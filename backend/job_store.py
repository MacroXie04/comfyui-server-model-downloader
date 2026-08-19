from __future__ import annotations

import base64
import contextlib
import fcntl
import json
import os
import stat
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

STATE_VERSION = 2
MAX_TERMINAL_HISTORY = 1_000
MAX_HISTORY_AGE_SECONDS = 90 * 24 * 60 * 60
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


class JobStoreError(ValueError):
    pass


class StateDirectoryLease:
    """A process-scoped advisory lease for one downloader state directory."""

    def __init__(self, state_directory: Path) -> None:
        self.path = Path(state_directory) / "instance.lock"
        self._descriptor = -1
        self.acquire()

    @property
    def acquired(self) -> bool:
        return self._descriptor >= 0

    def acquire(self) -> None:
        if self.acquired:
            return
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags, 0o600)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise JobStoreError("instance lock must be a regular file")
            os.fchmod(descriptor, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise JobStoreError(
                    "another Server Model Downloader process owns this state directory"
                ) from exc
            os.ftruncate(descriptor, 0)
            os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
            os.fsync(descriptor)
            self._descriptor = descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def release(self) -> None:
        if not self.acquired:
            return
        descriptor, self._descriptor = self._descriptor, -1
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __del__(self) -> None:  # pragma: no cover - best-effort interpreter cleanup
        with contextlib.suppress(Exception):
            self.release()


def _phase_for_status(status: object) -> str:
    return {
        "queued": "queued",
        "downloading": "downloading",
        "completed": "completed",
        "failed": "failed",
        "cancelled": "cancelled",
    }.get(status, "failed")


def migrate_job(row: Mapping[str, Any], source_version: int) -> dict[str, Any]:
    migrated = dict(row)
    if source_version not in (1, STATE_VERSION):
        raise JobStoreError("unsupported persisted job state")
    migrated.setdefault("phase", _phase_for_status(migrated.get("status")))
    migrated.setdefault("error_code", None)
    migrated.setdefault("attempt", 0)
    migrated.setdefault("max_attempts", 5)
    migrated.setdefault("cancel_requested", False)
    # Pre-release/internal state did not scope rows to an authenticated user.
    # Keep those rows available to the worker for safe recovery, but leave them
    # inaccessible through every user-facing history and mutation endpoint.
    migrated.setdefault("owner_binding", None)
    if source_version == 1 and migrated.get("status") == "failed":
        # v1 persisted raw exception strings, including OSError filenames.
        # Do not carry potentially sensitive legacy details into the public v2 API.
        migrated["error"] = "Previous download failed"
        migrated["error_code"] = "legacy_failure"
    return migrated


class PersistentJobStore:
    def __init__(self, state_directory: Path) -> None:
        self.state_directory = Path(state_directory)
        self.path = self.state_directory / "jobs.json"
        self.last_quarantine: Path | None = None

    def _quarantine(self) -> None:
        quarantine = self.state_directory / (
            f"jobs.corrupt-{int(time.time())}-{uuid.uuid4().hex[:8]}.json"
        )
        os.replace(self.path, quarantine)
        self.last_quarantine = quarantine

    def load(
        self, validator: Callable[[Mapping[str, Any]], None]
    ) -> tuple[dict[str, dict[str, Any]], bool]:
        try:
            mode = os.lstat(self.path).st_mode
        except FileNotFoundError:
            return {}, False
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise JobStoreError("jobs state must be a regular file")
        try:
            document = json.loads(self.path.read_text("utf-8"))
            if not isinstance(document, dict):
                raise JobStoreError("invalid persisted job state")
            version = document.get("version")
            if version not in (1, STATE_VERSION):
                raise JobStoreError("unsupported persisted job state")
            rows = document.get("jobs")
            if not isinstance(rows, list):
                raise JobStoreError("invalid persisted job state")
        except (OSError, json.JSONDecodeError, JobStoreError):
            self._quarantine()
            return {}, True

        changed = version != STATE_VERSION
        jobs: dict[str, dict[str, Any]] = {}
        for raw in rows:
            if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str):
                self._quarantine()
                return {}, True
            row = migrate_job(raw, int(version))
            if row != dict(raw):
                changed = True
            try:
                validator(row)
            except Exception:
                self._quarantine()
                return {}, True
            if row["id"] in jobs:
                self._quarantine()
                return {}, True
            jobs[row["id"]] = row
        return jobs, changed

    @staticmethod
    def prune(jobs: dict[str, dict[str, Any]], now: float | None = None) -> bool:
        now = time.time() if now is None else now
        cutoff = now - MAX_HISTORY_AGE_SECONDS
        remove: set[str] = set()
        terminal: list[dict[str, Any]] = []
        for job in jobs.values():
            if job.get("status") not in TERMINAL_STATUSES:
                continue
            terminal.append(job)
            completed = job.get("completed_at") or job.get("updated_at") or 0
            if isinstance(completed, (int, float)) and completed < cutoff:
                remove.add(job["id"])
        retained = [job for job in terminal if job["id"] not in remove]
        retained.sort(
            key=lambda item: (item.get("created_at", 0), item.get("id", "")),
            reverse=True,
        )
        remove.update(job["id"] for job in retained[MAX_TERMINAL_HISTORY:])
        for job_id in remove:
            jobs.pop(job_id, None)
        return bool(remove)

    def persist(self, jobs: dict[str, dict[str, Any]]) -> None:
        self.prune(jobs)
        document = {
            "version": STATE_VERSION,
            "jobs": sorted(jobs.values(), key=lambda item: item.get("created_at", 0)),
        }
        payload = json.dumps(
            document, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".jobs-", suffix=".tmp", dir=self.state_directory
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.path)
            directory_fd = os.open(
                self.state_directory,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary_name)
            raise


def encode_cursor(job: Mapping[str, Any]) -> str:
    raw = json.dumps(
        [float(job.get("created_at", 0)), str(job["id"])], separators=(",", ":")
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def decode_cursor(cursor: str) -> tuple[float, str]:
    if not isinstance(cursor, str) or not cursor or len(cursor) > 512:
        raise JobStoreError("invalid job history cursor")
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        value = json.loads(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise JobStoreError("invalid job history cursor") from exc
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not isinstance(value[0], (int, float))
        or isinstance(value[0], bool)
        or not isinstance(value[1], str)
    ):
        raise JobStoreError("invalid job history cursor")
    return float(value[0]), value[1]
