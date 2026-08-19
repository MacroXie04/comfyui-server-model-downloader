from __future__ import annotations

import asyncio
import contextlib
import ctypes
import errno
import fcntl
import hashlib
import os
import re
import shutil
import stat
import sys
import time
import uuid
from collections.abc import Mapping
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from .job_store import (
    JobStoreError,
    PersistentJobStore,
    StateDirectoryLease,
    decode_cursor,
    encode_cursor,
)
from .safetensors_check import SafeTensorsError, validate_safetensors_file
from .security import (
    SecurityError,
    TokenSigner,
    ensure_state_directory,
    require_public_dns,
    resolve_model_paths,
    validate_directory,
    validate_filename,
    validate_redirect_url,
    validate_sha256,
    validate_source_url,
)

RESERVE_BYTES = 20 * 1024**3
MAX_MODEL_BYTES = 10 * 1024**4
MAX_REDIRECTS = 8
CHUNK_BYTES = 4 * 1024 * 1024
MAX_DOWNLOAD_ATTEMPTS = 5
MAX_RETRY_DELAY_SECONDS = 300.0


class JobError(ValueError):
    pass


class DownloadCancelled(Exception):
    pass


class RetryableDownloadError(JobError):
    def __init__(
        self,
        message: str,
        *,
        error_code: str = "network_error",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.retry_after = retry_after


def _now() -> float:
    return time.time()


def _hash_file(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        return _hash_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _hash_descriptor(descriptor: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        raise SecurityError("model file is not a regular file")
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        block = os.read(descriptor, CHUNK_BYTES)
        if not block:
            break
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def _validate_safetensors_descriptor(descriptor: int) -> dict[str, Any]:
    """Validate the exact open inode rather than reopening a mutable name."""

    for root in (Path("/proc/self/fd"), Path("/dev/fd")):
        descriptor_path = root / str(descriptor)
        if descriptor_path.exists():
            return validate_safetensors_file(descriptor_path)
    raise SecurityError("open-file safetensors validation is unavailable")


def _link_validated_descriptor_no_replace(
    descriptor: int,
    directory_fd: int,
    destination_name: str,
) -> None:
    """Atomically publish the exact open inode without trusting its filename.

    This downloader is intentionally Linux-only. ``AT_EMPTY_PATH`` binds the
    hardlink to the open descriptor. Unprivileged systems that reject it use
    the Linux man-pages documented ``/proc/self/fd`` + ``AT_SYMLINK_FOLLOW``
    form, which remains descriptor-bound. There is deliberately no fallback
    to linking the mutable ``.part`` pathname.
    """

    if not sys.platform.startswith("linux"):
        raise SecurityError("descriptor-bound model publication requires Linux")
    if "/" in destination_name or destination_name in ("", ".", ".."):
        raise SecurityError("invalid publication filename")

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        linkat = libc.linkat
    except AttributeError as exc:
        raise SecurityError("descriptor-bound linkat is unavailable") from exc
    linkat.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    linkat.restype = ctypes.c_int
    encoded_destination = os.fsencode(destination_name)

    # Linux constants from <fcntl.h>. They are stable ABI values.
    at_fdcwd = -100
    at_symlink_follow = 0x400
    at_empty_path = 0x1000

    ctypes.set_errno(0)
    if linkat(descriptor, b"", directory_fd, encoded_destination, at_empty_path) == 0:
        return
    first_error = ctypes.get_errno()
    if first_error == errno.EEXIST:
        raise FileExistsError(first_error, os.strerror(first_error), destination_name)
    fallback_errors = {
        errno.EACCES,
        errno.EINVAL,
        errno.ENOENT,
        errno.ENOSYS,
        errno.EPERM,
    }
    if hasattr(errno, "EOPNOTSUPP"):
        fallback_errors.add(errno.EOPNOTSUPP)
    if first_error not in fallback_errors:
        raise OSError(first_error, os.strerror(first_error), destination_name)

    proc_descriptor = os.fsencode(f"/proc/self/fd/{descriptor}")
    ctypes.set_errno(0)
    if (
        linkat(
            at_fdcwd,
            proc_descriptor,
            directory_fd,
            encoded_destination,
            at_symlink_follow,
        )
        == 0
    ):
        return
    second_error = ctypes.get_errno()
    if second_error == errno.EEXIST:
        raise FileExistsError(second_error, os.strerror(second_error), destination_name)
    raise OSError(second_error, os.strerror(second_error), destination_name)


def _hash_partial(path: Path) -> tuple[hashlib._Hash, int]:  # type: ignore[name-defined]
    digest = hashlib.sha256()
    size = 0
    if path.exists():
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise SecurityError("partial download is not a regular file")
        with os.fdopen(descriptor, "rb") as stream:
            while True:
                block = stream.read(CHUNK_BYTES)
                if not block:
                    break
                digest.update(block)
                size += len(block)
    return digest, size


class JobManager:
    def __init__(
        self,
        models_root: Path,
        state_directory: Path,
        signer: TokenSigner | None = None,
        reserve_bytes: int = RESERVE_BYTES,
    ):
        self.models_root = Path(models_root)
        self.state_directory = ensure_state_directory(Path(state_directory))
        self.signer = signer or TokenSigner(self.state_directory)
        self.reserve_bytes = int(reserve_bytes)
        self.jobs_path = self.state_directory / "jobs.json"
        try:
            self._instance_lease = StateDirectoryLease(self.state_directory)
        except JobStoreError as exc:
            raise JobError(str(exc)) from exc
        self._store = PersistentJobStore(self.state_directory)
        self.locks_directory = ensure_state_directory(
            self.state_directory / "download-locks"
        )
        self._jobs: dict[str, dict[str, Any]] = {}
        self._worker_task: asyncio.Task[Any] | None = None
        self._wake: asyncio.Event | None = None
        self._stopping = False
        self._health_state = "ready"
        self._degraded_reason: str | None = None
        self._last_worker_error: str | None = None
        self._publication_checked = False
        try:
            self._load()
        except BaseException:
            self._instance_lease.release()
            raise

    def _load(self) -> None:
        try:
            loaded, changed = self._store.load(self._validate_job_record)
        except JobStoreError as exc:
            raise JobError(str(exc)) from exc
        self._jobs.update(loaded)
        for row in self._jobs.values():
            if row.get("status") == "downloading":
                if row.get("cancel_requested") is True:
                    row["status"] = "cancelled"
                    row["phase"] = "cancelled"
                    row["error"] = None
                    row["error_code"] = None
                else:
                    row["status"] = "queued"
                    row["phase"] = "queued"
                    row["error"] = (
                        "Recovered after service restart; partial file will resume"
                    )
                    row["error_code"] = None
                row["cancel_requested"] = False
                row["updated_at"] = _now()
                changed = True
        if changed:
            self._persist()

    @staticmethod
    def _validate_job_record(job: Mapping[str, Any]) -> None:
        if job.get("status") not in (
            "queued",
            "downloading",
            "completed",
            "failed",
            "cancelled",
        ):
            raise JobError("invalid persisted job status")
        provider = job.get("provider")
        source = validate_source_url(job.get("canonical_url"))
        if provider not in ("huggingface", "civitai") or source.provider != provider:
            raise JobError("invalid provider or canonical URL")
        directory = validate_directory(job.get("directory"))
        filename = validate_filename(job.get("filename"))
        if job.get("path") != f"{directory}/{filename}":
            raise JobError("invalid persisted destination path")
        revision = job.get("revision")
        if not isinstance(revision, str) or not revision:
            raise JobError("invalid persisted revision")
        if provider == "huggingface" and source.revision != revision:
            raise JobError("persisted Hugging Face revision mismatch")
        if provider == "civitai" and (
            str(source.version_id) != revision or source.file_id is None
        ):
            raise JobError("persisted Civitai version/file mismatch")
        size = job.get("size")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or size > MAX_MODEL_BYTES
        ):
            raise JobError("invalid persisted model size")
        validate_sha256(job.get("sha256"))
        phase = job.get("phase")
        if phase not in (
            "queued",
            "connecting",
            "downloading",
            "retrying",
            "hashing",
            "validating",
            "publishing",
            "cancelling",
            "completed",
            "failed",
            "cancelled",
        ):
            raise JobError("invalid persisted job phase")
        status = job.get("status")
        if status in ("queued", "completed", "failed", "cancelled") and phase != status:
            raise JobError("persisted job phase does not match status")
        if not isinstance(job.get("cancel_requested"), bool):
            raise JobError("invalid persisted cancellation state")
        error_code = job.get("error_code")
        if error_code is not None and (
            not isinstance(error_code, str)
            or not error_code
            or len(error_code) > 64
            or not re.fullmatch(r"[a-z][a-z0-9_]*", error_code)
        ):
            raise JobError("invalid persisted error code")
        error = job.get("error")
        if error is not None and (
            not isinstance(error, str)
            or len(error) > 2000
            or any(ord(character) < 32 for character in error)
            or "/" in error
            or "\\" in error
        ):
            raise JobError("invalid persisted error text")
        attempt = job.get("attempt")
        maximum = job.get("max_attempts")
        if (
            not isinstance(attempt, int)
            or isinstance(attempt, bool)
            or attempt < 0
            or not isinstance(maximum, int)
            or isinstance(maximum, bool)
            or maximum < 1
            or attempt > maximum
        ):
            raise JobError("invalid persisted retry state")
        owner_binding = job.get("owner_binding")
        if owner_binding is not None and (
            not isinstance(owner_binding, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{43}", owner_binding)
        ):
            raise JobError("invalid persisted job owner")

    def _persist(self) -> None:
        try:
            self._store.persist(self._jobs)
        except (OSError, TypeError, ValueError):
            self._health_state = "degraded"
            self._degraded_reason = "job state cannot be persisted"
            raise

    async def ensure_started(self) -> None:
        if self._health_state == "degraded":
            raise JobError(self._degraded_reason or "download worker is degraded")
        if not self._instance_lease.acquired:
            try:
                self._instance_lease.acquire()
            except JobStoreError as exc:
                self._health_state = "degraded"
                self._degraded_reason = str(exc)
                raise JobError(str(exc)) from exc
        if not self._publication_checked:
            try:
                await asyncio.to_thread(self._probe_publication_capability)
            except (OSError, SecurityError) as exc:
                self._health_state = "degraded"
                self._degraded_reason = f"safe publication is unavailable: {exc}"
                raise JobError(self._degraded_reason) from exc
            self._publication_checked = True
        self._stopping = False
        loop = asyncio.get_running_loop()
        if self._worker_task is None or self._worker_task.done():
            self._wake = asyncio.Event()
            self._worker_task = loop.create_task(
                self._supervisor(), name="server-model-downloader-supervisor"
            )
        self._health_state = "running"
        if any(job.get("status") == "queued" for job in self._jobs.values()):
            self._wake.set()

    async def stop(self) -> None:
        self._stopping = True
        task = self._worker_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._worker_task = None
        try:
            self._recover_interrupted_jobs("Paused during controlled shutdown")
            self._persist()
        finally:
            self._instance_lease.release()
        if self._health_state != "degraded":
            self._health_state = "stopped"

    def health(self) -> dict[str, Any]:
        task = self._worker_task
        return {
            "status": "degraded" if self._health_state == "degraded" else "ok",
            "state": self._health_state,
            "worker_running": bool(task is not None and not task.done()),
            "instance_lock_acquired": self._instance_lease.acquired,
            "queued_jobs": sum(
                job.get("status") == "queued" for job in self._jobs.values()
            ),
            "last_worker_error": self._last_worker_error,
            "reason": self._degraded_reason,
            "quarantined_state": self._store.last_quarantine is not None,
        }

    def _probe_publication_capability(self) -> None:
        probe_filename = f"smd_probe_{uuid.uuid4().hex}.safetensors"
        final_path, _ = resolve_model_paths(
            self.models_root, "diffusion_models", probe_filename
        )
        directory_fd = os.open(
            final_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        destination_name = f".smd-probe-link-{uuid.uuid4().hex}"
        source_name = f"{destination_name}.part"
        descriptor = -1
        try:
            descriptor = os.open(
                source_name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
            os.write(descriptor, b"publication-probe")
            os.fsync(descriptor)
            _link_validated_descriptor_no_replace(
                descriptor, directory_fd, destination_name
            )
            linked = os.stat(
                destination_name, dir_fd=directory_fd, follow_symlinks=False
            )
            if not os.path.samestat(os.fstat(descriptor), linked):
                raise SecurityError("descriptor-bound publication probe changed inode")
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            for name in (destination_name, source_name):
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(name, dir_fd=directory_fd)
            os.close(directory_fd)

    def _validated_payload(self, token: str, subject: str) -> dict[str, Any]:
        if not isinstance(subject, str) or not subject.strip():
            raise JobError("authenticated subject is required")
        payload = self.signer.verify_download(token, subject)
        provider = payload.get("provider")
        canonical_url = payload.get("canonical_url")
        source = validate_source_url(canonical_url)
        if source.provider != provider:
            raise JobError("signed provider does not match URL")
        directory = validate_directory(payload.get("directory"))
        filename = validate_filename(payload.get("filename"))
        if payload.get("path") != f"{directory}/{filename}":
            raise JobError("signed destination path is inconsistent")
        revision = payload.get("revision")
        if not isinstance(revision, str) or not revision:
            raise JobError("signed revision is missing")
        if provider == "huggingface" and source.revision != revision:
            raise JobError("Hugging Face URL is not pinned to the signed revision")
        if provider == "civitai" and str(source.version_id) != revision:
            raise JobError("Civitai URL does not match the signed model version")
        if provider == "civitai" and source.file_id is None:
            raise JobError("signed Civitai URL must identify an API-verified fileId")
        size = payload.get("size")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or size > MAX_MODEL_BYTES
        ):
            raise JobError("signed model size is invalid")
        payload["sha256"] = validate_sha256(payload.get("sha256"))
        payload["directory"] = directory
        payload["filename"] = filename
        payload["owner_binding"] = self.signer.subject_binding(subject)
        return payload

    def _check_capacity(
        self, payload: Mapping[str, Any], *, restart_from_zero: bool = False
    ) -> int:
        final_path, part_path = resolve_model_paths(
            self.models_root, payload["directory"], payload["filename"]
        )
        existing = os.lstat(part_path).st_size if part_path.exists() else 0
        remaining = (
            int(payload["size"])
            if restart_from_zero
            else max(0, int(payload["size"]) - existing)
        )
        free = shutil.disk_usage(final_path.parent).free
        # A 200 response to a Range request forces a truncate/restart. The
        # existing partial is still consuming disk at this point, so account
        # for the bytes that truncation will release without deleting it before
        # the safety decision is made.
        effective_free = free + existing if restart_from_zero else free
        other_reserved = 0
        seen_paths: set[str] = set()
        for active in self._jobs.values():
            if active.get("status") not in ("queued", "downloading"):
                continue
            if (
                active.get("path") == payload.get("path")
                or active.get("path") in seen_paths
            ):
                continue
            seen_paths.add(active["path"])
            other_reserved += self._remaining_for(active)
        if effective_free - remaining - other_reserved < self.reserve_bytes:
            raise JobError(
                "insufficient disk space: download must leave at least 20 GiB free"
            )
        return existing

    def _remaining_for(self, payload: Mapping[str, Any]) -> int:
        _, part_path = resolve_model_paths(
            self.models_root, payload["directory"], payload["filename"]
        )
        existing = os.lstat(part_path).st_size if part_path.exists() else 0
        return max(0, int(payload["size"]) - existing)

    async def _completed_file_is_present(self, job: Mapping[str, Any]) -> bool:
        try:
            final_path, _ = resolve_model_paths(
                self.models_root, job["directory"], job["filename"]
            )
            mode = os.lstat(final_path).st_mode
            if (
                not stat.S_ISREG(mode)
                or stat.S_ISLNK(mode)
                or os.lstat(final_path).st_size != job["size"]
            ):
                return False
            actual_hash, actual_size = await asyncio.to_thread(_hash_file, final_path)
            if actual_size != job["size"] or actual_hash != job["sha256"]:
                return False
            await asyncio.to_thread(validate_safetensors_file, final_path)
        except (FileNotFoundError, OSError, SecurityError, SafeTensorsError):
            return False
        return True

    async def create_jobs(
        self, tokens: list[str], license_confirmed: bool, subject: str
    ) -> list[dict[str, Any]]:
        if license_confirmed is not True:
            raise JobError("license_confirmed must be true before downloading")
        if not isinstance(tokens, list) or not tokens or len(tokens) > 20:
            raise JobError("download_tokens must contain 1 to 20 tokens")
        payloads = [self._validated_payload(token, subject) for token in tokens]
        owner_binding = self.signer.subject_binding(subject)
        returned: list[dict[str, Any] | None] = [None] * len(payloads)
        planned: dict[tuple[str, str], dict[str, Any]] = {}
        plan_positions: dict[tuple[str, str], list[int]] = {}
        planned_hash_by_path: dict[str, str] = {}
        for index, payload in enumerate(payloads):
            matching_active = None
            for existing_job in self._jobs.values():
                if existing_job.get("path") != payload["path"]:
                    continue
                if existing_job.get("status") in ("queued", "downloading"):
                    if existing_job.get("sha256") != payload["sha256"]:
                        raise JobError(
                            "an active download already targets this path with "
                            "a different SHA256"
                        )
                    if existing_job.get("owner_binding") != owner_binding:
                        raise JobError(
                            "an active download already targets this destination"
                        )
                    matching_active = existing_job
                    break
                if (
                    existing_job.get("status") == "completed"
                    and existing_job.get("sha256") == payload["sha256"]
                    and existing_job.get("owner_binding") == owner_binding
                    and await self._completed_file_is_present(existing_job)
                ):
                    matching_active = existing_job
                    break
            if matching_active is not None:
                returned[index] = self._public(matching_active)
                continue
            final_path, _ = resolve_model_paths(
                self.models_root, payload["directory"], payload["filename"]
            )
            if final_path.exists():
                raise JobError(
                    "destination already exists and is not a verified "
                    "completed download"
                )
            key = (payload["path"], payload["sha256"])
            prior_hash = planned_hash_by_path.get(payload["path"])
            if prior_hash is not None and prior_hash != payload["sha256"]:
                raise JobError(
                    "batch contains different SHA256 values for the same destination"
                )
            planned_hash_by_path[payload["path"]] = payload["sha256"]
            planned.setdefault(key, payload)
            plan_positions.setdefault(key, []).append(index)

        active_remaining = 0
        active_paths: set[str] = set()
        for active in self._jobs.values():
            if active.get("status") not in ("queued", "downloading"):
                continue
            if active["path"] in active_paths:
                continue
            active_paths.add(active["path"])
            active_remaining += self._remaining_for(active)
        planned_remaining = sum(
            self._remaining_for(payload) for payload in planned.values()
        )
        root_probe = next(iter(planned.values()), None)
        if root_probe is not None:
            final_probe, _ = resolve_model_paths(
                self.models_root, root_probe["directory"], root_probe["filename"]
            )
            free = shutil.disk_usage(final_probe.parent).free
            if free - active_remaining - planned_remaining < self.reserve_bytes:
                raise JobError(
                    "insufficient aggregate disk space: queued downloads must "
                    "leave at least 20 GiB free"
                )

        new_rows: list[dict[str, Any]] = []
        timestamp = _now()
        for key, payload in planned.items():
            partial_size = int(payload["size"]) - self._remaining_for(payload)
            job_id = uuid.uuid4().hex
            job = {
                "id": job_id,
                "provider": payload["provider"],
                "canonical_url": payload["canonical_url"],
                "revision": payload["revision"],
                "size": payload["size"],
                "sha256": payload["sha256"],
                "directory": payload["directory"],
                "filename": payload["filename"],
                "path": payload["path"],
                "license": payload.get("license"),
                "license_url": payload.get("license_url"),
                "status": "queued",
                "phase": "queued",
                "bytes_downloaded": partial_size,
                "created_at": timestamp,
                "updated_at": timestamp,
                "started_at": None,
                "completed_at": None,
                "error": None,
                "error_code": None,
                "attempt": 0,
                "max_attempts": MAX_DOWNLOAD_ATTEMPTS,
                "cancel_requested": False,
                "safetensors": None,
                "owner_binding": owner_binding,
            }
            new_rows.append(job)
            for position in plan_positions[key]:
                returned[position] = self._public(job)
        for job in new_rows:
            self._jobs[job["id"]] = job
        try:
            self._persist()
        except BaseException:
            for job in new_rows:
                self._jobs.pop(job["id"], None)
            raise
        await self.ensure_started()
        return [job for job in returned if job is not None]

    @staticmethod
    def _public(job: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(job)
        result.pop("owner_binding", None)
        size = int(result.get("size") or 0)
        downloaded = int(result.get("bytes_downloaded") or 0)
        result["progress"] = min(1.0, downloaded / size) if size else 0.0
        return result

    def list_jobs(self, subject: str) -> list[dict[str, Any]]:
        owner_binding = self.signer.subject_binding(subject)
        return [
            self._public(job)
            for job in sorted(
                (
                    job
                    for job in self._jobs.values()
                    if job.get("owner_binding") == owner_binding
                ),
                key=lambda item: item.get("created_at", 0),
                reverse=True,
            )
        ]

    def list_job_history(
        self, subject: str, *, cursor: str | None = None, limit: int = 50
    ) -> dict[str, Any]:
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 100
        ):
            raise JobError("history limit must be between 1 and 100")
        owner_binding = self.signer.subject_binding(subject)
        rows = sorted(
            (
                job
                for job in self._jobs.values()
                if job.get("owner_binding") == owner_binding
            ),
            key=lambda item: (item.get("created_at", 0), item.get("id", "")),
            reverse=True,
        )
        if cursor is not None:
            try:
                cursor_key = decode_cursor(cursor)
            except JobStoreError as exc:
                raise JobError(str(exc)) from exc
            rows = [
                row
                for row in rows
                if (float(row.get("created_at", 0)), str(row.get("id", "")))
                < cursor_key
            ]
        page = rows[:limit]
        next_cursor = encode_cursor(page[-1]) if len(rows) > limit and page else None
        return {
            "jobs": [self._public(job) for job in page],
            "next_cursor": next_cursor,
        }

    def _owned_job(self, job_id: str, subject: str) -> dict[str, Any]:
        try:
            job = self._jobs[job_id]
        except KeyError as exc:
            raise KeyError("job not found") from exc
        if job.get("owner_binding") != self.signer.subject_binding(subject):
            raise KeyError("job not found")
        return job

    def get_job(self, job_id: str, subject: str) -> dict[str, Any]:
        return self._public(self._owned_job(job_id, subject))

    def cancel_job(self, job_id: str, subject: str) -> dict[str, Any]:
        job = self._owned_job(job_id, subject)
        if job["status"] == "queued":
            job["status"] = "cancelled"
            job["phase"] = "cancelled"
            job["error"] = None
            job["error_code"] = None
        elif job["status"] == "downloading":
            job["cancel_requested"] = True
            job["phase"] = "cancelling"
        job["updated_at"] = _now()
        self._persist()
        if self._wake:
            self._wake.set()
        return self._public(job)

    async def discard_partial(self, job_id: str, subject: str) -> dict[str, Any]:
        job = self._owned_job(job_id, subject)
        if job.get("status") not in ("failed", "cancelled"):
            raise JobError(
                "partial files may only be discarded for terminal failed jobs"
            )
        lock_descriptor = await self._acquire_destination_lock(job)
        try:
            final_path, part_path = resolve_model_paths(
                self.models_root, job["directory"], job["filename"]
            )
            if final_path.exists():
                raise JobError("partial cannot be discarded after final publication")
            directory_fd = os.open(
                final_path.parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                try:
                    metadata = os.stat(
                        part_path.name, dir_fd=directory_fd, follow_symlinks=False
                    )
                except FileNotFoundError:
                    metadata = None
                if metadata is not None:
                    if not stat.S_ISREG(metadata.st_mode):
                        raise SecurityError("partial download is not a regular file")
                    os.unlink(part_path.name, dir_fd=directory_fd)
                    os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            job["bytes_downloaded"] = 0
            job["updated_at"] = _now()
            self._persist()
            return self._public(job)
        finally:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)

    def _next_queued(self) -> dict[str, Any] | None:
        queued = [job for job in self._jobs.values() if job.get("status") == "queued"]
        return (
            min(queued, key=lambda item: item.get("created_at", 0)) if queued else None
        )

    async def _worker(self) -> None:
        assert self._wake is not None
        while True:
            job = self._next_queued()
            if job is None:
                self._wake.clear()
                job = self._next_queued()
                if job is None:
                    await self._wake.wait()
                    continue
            await self._run_job(job)

    async def _supervisor(self) -> None:
        while not self._stopping:
            try:
                await self._worker()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_worker_error = type(exc).__name__
                self._recover_interrupted_jobs(
                    "Recovered after worker failure; partial file will resume"
                )
                try:
                    self._persist()
                except OSError:
                    self._health_state = "degraded"
                    self._degraded_reason = (
                        "worker failed and recovery state could not be persisted"
                    )
                    return
                if not self._stopping:
                    await asyncio.sleep(1)

    def _recover_interrupted_jobs(self, message: str) -> None:
        for job in self._jobs.values():
            if job.get("status") != "downloading":
                continue
            if job.get("cancel_requested"):
                job["status"] = "cancelled"
                job["phase"] = "cancelled"
                job["error"] = None
                job["error_code"] = None
            else:
                job["status"] = "queued"
                job["phase"] = "queued"
                job["error"] = message
                job["error_code"] = None
            job["cancel_requested"] = False
            job["updated_at"] = _now()

    @staticmethod
    def _retryable_exception(exc: Exception) -> RetryableDownloadError | None:
        if isinstance(exc, RetryableDownloadError):
            return exc
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return RetryableDownloadError("download timed out", error_code="timeout")
        if isinstance(exc, ConnectionError):
            return RetryableDownloadError(
                "download connection failed", error_code="network_error"
            )
        try:
            import aiohttp

            client_error = getattr(aiohttp, "ClientError", None)
            if isinstance(client_error, type) and isinstance(exc, client_error):
                return RetryableDownloadError(
                    "provider connection failed", error_code="network_error"
                )
        except ImportError:
            pass
        return None

    @staticmethod
    def _failure_code(exc: Exception) -> str:
        if isinstance(exc, RetryableDownloadError):
            return exc.error_code
        if isinstance(exc, SecurityError):
            return "security_error"
        if isinstance(exc, SafeTensorsError):
            return "integrity_error"
        if isinstance(exc, OSError):
            return "filesystem_error"
        message = str(exc).lower()
        if "disk space" in message:
            return "insufficient_space"
        if any(
            word in message for word in ("sha256", "size", "safetensors", "integrity")
        ):
            return "integrity_error"
        if "http" in message or "provider" in message:
            return "upstream_http_error"
        return "job_error"

    @staticmethod
    def _retry_delay(attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(MAX_RETRY_DELAY_SECONDS, max(0.0, retry_after))
        return min(60.0, float(2 ** max(0, attempt - 1)))

    @staticmethod
    async def _sleep_before_retry(job: Mapping[str, Any], delay: float) -> None:
        deadline = time.monotonic() + delay
        while True:
            if job.get("cancel_requested"):
                raise DownloadCancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.25, remaining))

    async def _run_job(self, job: dict[str, Any]) -> None:
        job["status"] = "downloading"
        job["cancel_requested"] = False
        job["started_at"] = job.get("started_at") or _now()
        job["updated_at"] = _now()
        job["error"] = None
        job["error_code"] = None
        job.setdefault("attempt", 0)
        job.setdefault("max_attempts", MAX_DOWNLOAD_ATTEMPTS)
        self._persist()
        try:
            first_attempt = min(int(job["attempt"]) + 1, int(job["max_attempts"]))
            for attempt in range(first_attempt, int(job["max_attempts"]) + 1):
                job["attempt"] = attempt
                job["phase"] = "connecting"
                job["error"] = None
                job["error_code"] = None
                job["updated_at"] = _now()
                self._persist()
                try:
                    await self._download(job)
                    break
                except DownloadCancelled:
                    raise
                except Exception as exc:
                    retryable = self._retryable_exception(exc)
                    if retryable is None:
                        raise
                    if attempt >= int(job["max_attempts"]):
                        if retryable is exc:
                            raise
                        raise retryable from exc
                    job["phase"] = "retrying"
                    job["error"] = str(retryable)[:2000]
                    job["error_code"] = retryable.error_code
                    job["updated_at"] = _now()
                    self._persist()
                    await self._sleep_before_retry(
                        job, self._retry_delay(attempt, retryable.retry_after)
                    )
        except DownloadCancelled:
            job["status"] = "cancelled"
            job["phase"] = "cancelled"
            job["error"] = None
            job["error_code"] = None
            job["cancel_requested"] = False
        except asyncio.CancelledError:
            job["status"] = "queued"
            job["phase"] = "queued"
            job["error"] = "Paused during controlled shutdown"
            job["error_code"] = None
            job["cancel_requested"] = False
            job["updated_at"] = _now()
            self._persist()
            raise
        except asyncio.TimeoutError:
            job["status"] = "failed"
            job["phase"] = "failed"
            job["error"] = "download timed out"
            job["error_code"] = "timeout"
        except OSError:
            job["status"] = "failed"
            job["phase"] = "failed"
            job["error"] = "filesystem operation failed"
            job["error_code"] = "filesystem_error"
        except SafeTensorsError:
            job["status"] = "failed"
            job["phase"] = "failed"
            job["error"] = "safetensors validation failed"
            job["error_code"] = "integrity_error"
        except (JobError, SecurityError) as exc:
            job["status"] = "failed"
            job["phase"] = "failed"
            job["error"] = str(exc)[:2000]
            job["error_code"] = self._failure_code(exc)
        except Exception:  # keep the persistent worker alive on provider failures
            job["status"] = "failed"
            job["phase"] = "failed"
            job["error"] = "unexpected download failure"
            job["error_code"] = "unexpected_error"
        job["updated_at"] = _now()
        self._persist()

    @staticmethod
    def _raise_if_cancelled(job: dict[str, Any]) -> None:
        if job.get("cancel_requested"):
            job["phase"] = "cancelling"
            raise DownloadCancelled()

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        if not value:
            return None
        stripped = value.strip()
        if stripped.isdigit():
            return min(MAX_RETRY_DELAY_SECONDS, float(stripped))
        try:
            parsed = parsedate_to_datetime(stripped)
            delay = parsed.timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return None
        return min(MAX_RETRY_DELAY_SECONDS, max(0.0, delay))

    @staticmethod
    def _download_headers(provider: str, host: str, offset: int) -> dict[str, str]:
        if provider not in ("huggingface", "civitai"):
            raise SecurityError("unknown download provider")
        headers = {
            "Accept": "application/octet-stream",
            "Accept-Encoding": "identity",
            "User-Agent": "ComfyUI-ServerModelDownloader/1.0",
        }
        if offset:
            headers["Range"] = f"bytes={offset}-"
        origin_host = "huggingface.co" if provider == "huggingface" else "civitai.com"
        if host == origin_host:
            if provider == "huggingface":
                token = os.environ.get("HF_TOKEN") or os.environ.get(
                    "HUGGING_FACE_HUB_TOKEN"
                )
            else:
                token = os.environ.get("CIVITAI_API_TOKEN")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        return headers

    async def _open_response(
        self, session: Any, job: Mapping[str, Any], offset: int
    ) -> Any:
        current = job["canonical_url"]
        provider = job["provider"]
        for _ in range(MAX_REDIRECTS + 1):
            host = validate_redirect_url(current, provider)
            await require_public_dns(host)
            response = await session.get(
                current,
                headers=self._download_headers(provider, host, offset),
                allow_redirects=False,
            )
            if response.status in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                response.release()
                if not location:
                    raise JobError("download redirect is missing Location")
                current = urljoin(current, location)
                validate_redirect_url(current, provider)
                continue
            return response
        raise JobError("too many download redirects")

    async def _acquire_destination_lock(self, job: Mapping[str, Any]) -> int:
        lock_name = hashlib.sha256(str(job["path"]).encode("utf-8")).hexdigest()
        lock_path = self.locks_directory / f"{lock_name}.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            lock_stat = os.fstat(descriptor)
            if not stat.S_ISREG(lock_stat.st_mode):
                raise SecurityError("download lock is not a regular file")
            os.fchmod(descriptor, 0o600)
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return descriptor
                except BlockingIOError:
                    if job.get("cancel_requested"):
                        raise DownloadCancelled() from None
                    await asyncio.sleep(0.25)
        except BaseException:
            os.close(descriptor)
            raise

    async def _download(self, job: dict[str, Any]) -> None:
        # The lock covers every read/write of the resumable .part path and the
        # descriptor-bound publication. It prevents another process running
        # this extension from truncating a hardlink while completion is being
        # committed.
        lock_descriptor = await self._acquire_destination_lock(job)
        try:
            await self._download_locked(job)
        finally:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)

    async def _download_locked(self, job: dict[str, Any]) -> None:
        try:
            import aiohttp
        except ImportError as exc:  # pragma: no cover - ComfyUI always includes aiohttp
            raise JobError(
                "aiohttp is required by ComfyUI Server Model Downloader"
            ) from exc

        final_path, part_path = resolve_model_paths(
            self.models_root, job["directory"], job["filename"]
        )
        if final_path.exists():
            self._raise_if_cancelled(job)
            job["phase"] = "hashing"
            actual_hash, actual_size = await asyncio.to_thread(_hash_file, final_path)
            if actual_hash != job["sha256"]:
                raise JobError("destination already exists with a different SHA256")
            if actual_size != job["size"]:
                raise JobError("destination already exists with a different size")
            self._raise_if_cancelled(job)
            job["phase"] = "validating"
            validation = await asyncio.to_thread(validate_safetensors_file, final_path)
            self._raise_if_cancelled(job)
            if part_path.exists():
                final_stat = os.lstat(final_path)
                part_stat = os.lstat(part_path)
                # Only remove a legacy hardlink alias to the already-verified
                # final. A separate partial/racer is never deleted here. The
                # per-target flock serializes this check and unlink across all
                # instances of this extension.
                if os.path.samestat(final_stat, part_stat):
                    part_path.unlink()
            job["bytes_downloaded"] = actual_size
            job["safetensors"] = validation
            job["status"] = "completed"
            job["phase"] = "completed"
            job["completed_at"] = _now()
            return

        if part_path.exists() and part_path.stat().st_size > job["size"]:
            raise JobError(
                "partial file is larger than signed model size; move or remove "
                "it manually"
            )
        digest, offset = await asyncio.to_thread(_hash_partial, part_path)
        job["bytes_downloaded"] = offset
        self._check_capacity(job)

        timeout = aiohttp.ClientTimeout(
            total=None, connect=30, sock_connect=30, sock_read=120
        )
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
            response = await self._open_response(session, job, offset)
            try:
                if response.status == 429:
                    raise RetryableDownloadError(
                        "provider rate limited the download",
                        error_code="upstream_rate_limited",
                        retry_after=self._parse_retry_after(
                            response.headers.get("Retry-After")
                        ),
                    )
                if 500 <= response.status <= 599:
                    raise RetryableDownloadError(
                        f"provider temporarily failed with HTTP {response.status}",
                        error_code="upstream_server_error",
                        retry_after=self._parse_retry_after(
                            response.headers.get("Retry-After")
                        ),
                    )
                if response.status == 416 and offset == job["size"]:
                    pass
                elif response.status not in (200, 206):
                    raise JobError(f"download failed with HTTP {response.status}")
                else:
                    append = offset > 0 and response.status == 206
                    if append:
                        content_range = response.headers.get("Content-Range", "")
                        match = re.fullmatch(
                            r"bytes ([0-9]+)-([0-9]+)/([0-9]+)", content_range
                        )
                        if not match or int(match.group(1)) != offset:
                            raise JobError(
                                "provider returned an invalid resume Content-Range"
                            )
                        remote_total = int(match.group(3))
                    else:
                        offset = 0
                        digest = hashlib.sha256()
                        remote_total = None
                        content_length = response.headers.get("Content-Length")
                        if content_length:
                            try:
                                remote_total = int(content_length)
                            except ValueError as exc:
                                raise JobError(
                                    "provider returned an invalid Content-Length"
                                ) from exc
                    if remote_total is not None and remote_total != job["size"]:
                        # Civitai reports sizeKB in its API, which can differ by a
                        # few bytes after decimal rounding. SHA256 remains binding.
                        if (
                            job["provider"] == "civitai"
                            and abs(remote_total - job["size"]) <= 2048
                        ):
                            job["size"] = remote_total
                        else:
                            raise JobError(
                                "download size does not match signed provider metadata"
                            )
                    self._check_capacity(job, restart_from_zero=not append)
                    final_path, part_path = resolve_model_paths(
                        self.models_root, job["directory"], job["filename"]
                    )
                    flags = os.O_WRONLY | os.O_CREAT
                    flags |= os.O_APPEND if append else os.O_TRUNC
                    if hasattr(os, "O_NOFOLLOW"):
                        flags |= os.O_NOFOLLOW
                    descriptor = os.open(part_path, flags, 0o640)
                    try:
                        mode = os.fstat(descriptor).st_mode
                        if not stat.S_ISREG(mode):
                            raise SecurityError(
                                "partial download is not a regular file"
                            )
                        with os.fdopen(descriptor, "ab" if append else "wb") as stream:
                            descriptor = -1
                            job["phase"] = "downloading"
                            last_persisted = time.monotonic()
                            async for chunk in response.content.iter_chunked(
                                CHUNK_BYTES
                            ):
                                if job.get("cancel_requested"):
                                    raise DownloadCancelled()
                                stream.write(chunk)
                                digest.update(chunk)
                                offset += len(chunk)
                                if offset > job["size"]:
                                    raise JobError(
                                        "download exceeded signed model size"
                                    )
                                job["bytes_downloaded"] = offset
                                now = time.monotonic()
                                if now - last_persisted >= 2:
                                    job["updated_at"] = _now()
                                    self._persist()
                                    last_persisted = now
                            stream.flush()
                            os.fsync(stream.fileno())
                    finally:
                        if descriptor >= 0:
                            os.close(descriptor)
            finally:
                response.release()

        self._raise_if_cancelled(job)
        if offset != job["size"]:
            raise JobError(f"download is incomplete ({offset} of {job['size']} bytes)")
        if digest.hexdigest().lower() != job["sha256"]:
            raise JobError("downloaded file failed SHA256 verification")

        # Re-resolve immediately before validation and publication. From this
        # point onward every integrity operation is bound to one O_NOFOLLOW
        # descriptor; the mutable .part name is never used as a link source.
        final_path, checked_part_path = resolve_model_paths(
            self.models_root, job["directory"], job["filename"]
        )
        if checked_part_path != part_path:
            raise SecurityError("partial path changed during download")
        directory_flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        directory_fd = os.open(final_path.parent, directory_flags)
        part_descriptor = -1
        try:
            part_descriptor = os.open(
                part_path.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            source_stat = os.fstat(part_descriptor)
            if not stat.S_ISREG(source_stat.st_mode):
                raise SecurityError("partial download is not a regular file")

            self._raise_if_cancelled(job)
            job["phase"] = "hashing"
            job["updated_at"] = _now()
            self._persist()
            actual_hash, actual_size = await asyncio.to_thread(
                _hash_descriptor, part_descriptor
            )
            if actual_size != job["size"] or actual_hash != job["sha256"]:
                raise JobError("on-disk partial failed final SHA256 verification")
            self._raise_if_cancelled(job)
            job["phase"] = "validating"
            job["updated_at"] = _now()
            self._persist()
            validation = await asyncio.to_thread(
                _validate_safetensors_descriptor, part_descriptor
            )
            self._raise_if_cancelled(job)
            # Bind the parser result to the same inode after parsing completes.
            job["phase"] = "hashing"
            final_hash, final_size = await asyncio.to_thread(
                _hash_descriptor, part_descriptor
            )
            if final_size != job["size"] or final_hash != job["sha256"]:
                raise JobError("partial inode changed during validation")

            self._raise_if_cancelled(job)
            job["phase"] = "publishing"
            job["updated_at"] = _now()
            self._persist()
            try:
                _link_validated_descriptor_no_replace(
                    part_descriptor, directory_fd, final_path.name
                )
            except FileExistsError as exc:
                raise JobError(
                    "destination appeared while the download was running"
                ) from exc
            os.fsync(directory_fd)
            current_final_stat = os.stat(
                final_path.name, dir_fd=directory_fd, follow_symlinks=False
            )
            if not os.path.samestat(source_stat, current_final_stat):
                # A racing replacement is never deleted. The exact validated
                # inode was linked atomically, so no unvalidated file was
                # introduced by this worker.
                raise JobError("destination changed immediately after publication")
            try:
                current_part_stat = os.stat(
                    part_path.name, dir_fd=directory_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                raise JobError("partial name disappeared during publication") from None
            if not os.path.samestat(source_stat, current_part_stat):
                # Never delete a replacement supplied outside this locked
                # downloader. The published final still refers to the exact
                # verified descriptor and will be recognized on retry.
                raise JobError("partial name changed during publication")
            os.unlink(part_path.name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        finally:
            if part_descriptor >= 0:
                os.close(part_descriptor)
            os.close(directory_fd)
        job["bytes_downloaded"] = job["size"]
        job["safetensors"] = validation
        job["status"] = "completed"
        job["phase"] = "completed"
        job["completed_at"] = _now()
