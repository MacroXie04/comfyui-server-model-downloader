from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import sys
import types
from collections import namedtuple
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from backend import jobs as jobs_module
from backend.jobs import (
    DownloadCancelled,
    JobError,
    JobManager,
    RetryableDownloadError,
)
from backend.safetensors_check import SafeTensorsError
from backend.security import SecurityError, TokenSigner

SUBJECT = "user-1"


def _run(coro):  # noqa: ANN001, ANN202
    return asyncio.run(coro)


def _manager(
    tmp_path: Path,
    *,
    reserve_bytes: int = 0,
    signer: TokenSigner | None = None,
) -> JobManager:
    state = tmp_path / "state"
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    signer = signer or TokenSigner(state)
    return JobManager(
        models,
        state,
        signer=signer,
        reserve_bytes=reserve_bytes,
    )


def _payload(
    *,
    filename: str = "model.safetensors",
    directory: str = "diffusion_models",
    size: int = 100,
    sha256: str = "a" * 64,
    repo: str = "org/repo",
    revision: str = "b" * 40,
) -> dict[str, Any]:
    return {
        "provider": "huggingface",
        "canonical_url": (
            f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
        ),
        "revision": revision,
        "size": size,
        "sha256": sha256,
        "directory": directory,
        "filename": filename,
        "path": f"{directory}/{filename}",
        "license": "apache-2.0",
        "license_url": f"https://huggingface.co/{repo}/blob/{revision}/LICENSE",
    }


def _token(signer: TokenSigner, **kwargs: Any) -> str:
    token, _ = signer.sign_download(_payload(**kwargs), SUBJECT)
    return token


def _disk_usage(free: int):  # noqa: ANN202
    usage = namedtuple("usage", "total used free")
    return usage(free * 2, free, free)


def test_batch_capacity_is_aggregated_and_failure_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    signer = TokenSigner(tmp_path / "state")
    manager = _manager(tmp_path, reserve_bytes=100, signer=signer)
    monkeypatch.setattr(
        jobs_module.shutil, "disk_usage", lambda path: _disk_usage(1_000)
    )
    tokens = [
        _token(signer, filename="one.safetensors", size=500, sha256="1" * 64),
        _token(signer, filename="two.safetensors", size=500, sha256="2" * 64),
    ]
    with pytest.raises(JobError, match="disk space"):
        _run(manager.create_jobs(tokens, True, SUBJECT))
    assert manager.list_jobs(SUBJECT) == []
    if manager.jobs_path.exists():
        assert json.loads(manager.jobs_path.read_text())["jobs"] == []


def test_destination_lock_serializes_waiters(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    job = _job_for_bytes(manager, b"lock-test")

    async def exercise() -> None:
        first_descriptor = await manager._acquire_destination_lock(job)
        second_waiter = asyncio.create_task(manager._acquire_destination_lock(job))
        await asyncio.sleep(0.05)
        assert not second_waiter.done()
        fcntl.flock(first_descriptor, fcntl.LOCK_UN)
        os.close(first_descriptor)
        second_descriptor = await asyncio.wait_for(second_waiter, timeout=1)
        fcntl.flock(second_descriptor, fcntl.LOCK_UN)
        os.close(second_descriptor)

    _run(exercise())


def test_state_directory_lease_rejects_second_manager(tmp_path: Path) -> None:
    first = _manager(tmp_path)
    with pytest.raises(JobError, match="another Server Model Downloader"):
        _manager(tmp_path)
    first._instance_lease.release()


def test_active_queued_downloads_are_included_in_capacity_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    signer = TokenSigner(tmp_path / "state")
    manager = _manager(tmp_path, reserve_bytes=100, signer=signer)
    monkeypatch.setattr(
        jobs_module.shutil, "disk_usage", lambda path: _disk_usage(1_000)
    )
    [first] = _run(
        manager.create_jobs(
            [_token(signer, filename="one.safetensors", size=500, sha256="1" * 64)],
            True,
            SUBJECT,
        )
    )
    assert first["status"] == "queued"
    with pytest.raises(JobError, match="disk space"):
        _run(
            manager.create_jobs(
                [_token(signer, filename="two.safetensors", size=500, sha256="2" * 64)],
                True,
                SUBJECT,
            )
        )
    assert [job["filename"] for job in manager.list_jobs(SUBJECT)] == [
        "one.safetensors"
    ]


def test_batch_validation_failure_does_not_leave_earlier_jobs_in_memory_or_state(
    tmp_path: Path,
) -> None:
    signer = TokenSigner(tmp_path / "state")
    manager = _manager(tmp_path, signer=signer)
    good = _token(signer, filename="good.safetensors", size=10)
    with pytest.raises(SecurityError, match="token"):
        _run(manager.create_jobs([good, "not-a-signed-token"], True, SUBJECT))
    assert manager.list_jobs(SUBJECT) == []
    assert not manager.jobs_path.exists()


def test_download_token_is_bound_to_authenticated_subject(tmp_path: Path) -> None:
    signer = TokenSigner(tmp_path / "state")
    manager = _manager(tmp_path, signer=signer)
    token = _token(signer)
    with pytest.raises(SecurityError, match="subject|token"):
        _run(manager.create_jobs([token], True, "different-user"))
    assert manager.list_jobs(SUBJECT) == []


def test_job_history_and_mutations_are_scoped_to_authenticated_subject(
    tmp_path: Path,
) -> None:
    signer = TokenSigner(tmp_path / "state")
    manager = _manager(tmp_path, signer=signer)
    [created] = _run(manager.create_jobs([_token(signer)], True, SUBJECT))
    other_subject = "user-2"

    assert manager.list_jobs(other_subject) == []
    assert manager.list_job_history(other_subject)["jobs"] == []
    for operation in (
        lambda: manager.get_job(created["id"], other_subject),
        lambda: manager.cancel_job(created["id"], other_subject),
    ):
        with pytest.raises(KeyError, match="not found"):
            operation()
    with pytest.raises(KeyError, match="not found"):
        _run(manager.discard_partial(created["id"], other_subject))

    assert manager.get_job(created["id"], SUBJECT)["id"] == created["id"]
    assert "owner_binding" not in manager.get_job(created["id"], SUBJECT)


def test_same_destination_with_different_hash_is_a_hard_conflict(
    tmp_path: Path,
) -> None:
    signer = TokenSigner(tmp_path / "state")
    manager = _manager(tmp_path, signer=signer)
    _run(manager.create_jobs([_token(signer, sha256="1" * 64)], True, SUBJECT))
    before = manager.list_jobs(SUBJECT)
    with pytest.raises(JobError, match="destination|path|conflict"):
        _run(manager.create_jobs([_token(signer, sha256="2" * 64)], True, SUBJECT))
    assert manager.list_jobs(SUBJECT) == before


def test_completed_job_with_missing_final_file_is_requeued_for_resume(
    tmp_path: Path,
) -> None:
    signer = TokenSigner(tmp_path / "state")
    manager = _manager(tmp_path, signer=signer)
    token = _token(signer, size=10)
    [created] = _run(manager.create_jobs([token], True, SUBJECT))
    internal = manager._jobs[created["id"]]
    internal["status"] = "completed"
    internal["bytes_downloaded"] = 10
    internal["completed_at"] = 123.0
    manager._persist()
    assert not (tmp_path / "models" / "diffusion_models" / "model.safetensors").exists()

    [recovered] = _run(manager.create_jobs([token], True, SUBJECT))
    assert recovered["status"] == "queued"
    assert recovered["completed_at"] is None
    # Keeping the stale completed row as immutable history is acceptable, but
    # there must be exactly one new active job and it must not be deduplicated
    # to the missing completed artifact.
    active = [
        job
        for job in manager.list_jobs(SUBJECT)
        if job["status"] in ("queued", "downloading")
    ]
    assert [job["id"] for job in active] == [recovered["id"]]


def test_completed_job_with_same_size_wrong_hash_is_not_reused_or_overwritten(
    tmp_path: Path,
) -> None:
    signer = TokenSigner(tmp_path / "state")
    manager = _manager(tmp_path, signer=signer)
    expected = b"expected!!"
    token = _token(
        signer,
        size=len(expected),
        sha256=hashlib.sha256(expected).hexdigest(),
    )
    [created] = _run(manager.create_jobs([token], True, SUBJECT))
    internal = manager._jobs[created["id"]]
    internal["status"] = "completed"
    internal["bytes_downloaded"] = len(expected)
    internal["completed_at"] = 123.0
    final = tmp_path / "models" / "diffusion_models" / "model.safetensors"
    final.write_bytes(b"tampered!!")
    assert final.stat().st_size == len(expected)
    with pytest.raises(JobError, match="destination already exists"):
        _run(manager.create_jobs([token], True, SUBJECT))
    assert final.read_bytes() == b"tampered!!"


def test_existing_partial_reduces_only_that_jobs_remaining_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    signer = TokenSigner(tmp_path / "state")
    manager = _manager(tmp_path, reserve_bytes=100, signer=signer)
    target = tmp_path / "models" / "diffusion_models"
    target.mkdir(parents=True)
    (target / "one.safetensors.part").write_bytes(b"x" * 400)
    monkeypatch.setattr(jobs_module.shutil, "disk_usage", lambda path: _disk_usage(700))
    # one needs only 100 additional bytes; two needs 400; exactly 100 bytes
    # remain reserved after both jobs, so this batch is allowed.
    created = _run(
        manager.create_jobs(
            [
                _token(signer, filename="one.safetensors", size=500, sha256="1" * 64),
                _token(signer, filename="two.safetensors", size=400, sha256="2" * 64),
            ],
            True,
            SUBJECT,
        )
    )
    assert [job["bytes_downloaded"] for job in created] == [400, 0]


class _Content:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        after_stream: Callable[[], None] | None = None,
    ) -> None:
        self.chunks = chunks
        self.after_stream = after_stream

    async def iter_chunked(self, size: int):  # noqa: ANN201
        for chunk in self.chunks:
            yield chunk
        if self.after_stream:
            self.after_stream()


class _Response:
    def __init__(
        self,
        status: int,
        *,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
        after_stream: Callable[[], None] | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self.content = _Content(chunks or [], after_stream=after_stream)
        self.released = False

    def release(self) -> None:
        self.released = True


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self):  # noqa: ANN204
        return self

    async def __aexit__(self, exc_type, exc, traceback):  # noqa: ANN001, ANN204
        return False

    async def get(self, url: str, **kwargs):  # noqa: ANN201
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def _install_aiohttp(monkeypatch: pytest.MonkeyPatch, session: _Session) -> None:
    module = types.ModuleType("aiohttp")
    module.ClientTimeout = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    module.ClientSession = lambda **kwargs: session  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aiohttp", module)


def _job_for_bytes(
    manager: JobManager,
    data: bytes,
    *,
    filename: str = "model.safetensors",
) -> dict[str, Any]:
    payload = _payload(
        filename=filename,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    timestamp = 1.0
    return {
        "id": "job",
        **payload,
        "status": "queued",
        "bytes_downloaded": 0,
        "created_at": timestamp,
        "updated_at": timestamp,
        "started_at": None,
        "completed_at": None,
        "error": None,
        "cancel_requested": False,
        "safetensors": None,
        "owner_binding": manager.signer.subject_binding(SUBJECT),
    }


@pytest.fixture
def accept_safetensors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        jobs_module,
        "validate_safetensors_file",
        lambda path: {"size": Path(path).stat().st_size, "tensor_count": 1},
    )


@pytest.fixture
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    async def accepted(host: str, port: int = 443) -> None:
        return None

    monkeypatch.setattr(jobs_module, "require_public_dns", accepted)


@pytest.fixture(autouse=True)
def portable_descriptor_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep host tests runnable on macOS; production fails closed off Linux."""

    if sys.platform.startswith("linux"):
        return

    def link_for_host_tests(
        descriptor: int, directory_fd: int, destination_name: str
    ) -> None:
        source_name = f"{destination_name}.part"
        source_stat = os.stat(source_name, dir_fd=directory_fd, follow_symlinks=False)
        if not os.path.samestat(os.fstat(descriptor), source_stat):
            raise SecurityError("test source name no longer identifies descriptor")
        os.link(
            source_name,
            destination_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )

    monkeypatch.setattr(
        jobs_module,
        "_link_validated_descriptor_no_replace",
        link_for_host_tests,
    )


def test_resume_206_requires_exact_content_range_and_appends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accept_safetensors: None,
    public_dns: None,
) -> None:
    manager = _manager(tmp_path)
    data = b"abcdefghij"
    job = _job_for_bytes(manager, data)
    part = tmp_path / "models" / "diffusion_models" / "model.safetensors.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(data[:4])
    response = _Response(
        206,
        headers={"Content-Range": "bytes 4-9/10", "Content-Length": "6"},
        chunks=[data[4:7], data[7:]],
    )
    session = _Session([response])
    _install_aiohttp(monkeypatch, session)
    _run(manager._download(job))
    final = part.with_name("model.safetensors")
    assert final.read_bytes() == data
    assert not part.exists()
    assert job["status"] == "completed"
    assert session.calls[0][1]["headers"]["Range"] == "bytes=4-"
    assert response.released


@pytest.mark.parametrize(
    "content_range",
    ["", "bytes 3-9/10", "bytes 4-9/11", "bytes */10", "garbage"],
)
def test_resume_206_rejects_invalid_content_range_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,
    content_range: str,
) -> None:
    manager = _manager(tmp_path)
    data = b"abcdefghij"
    job = _job_for_bytes(manager, data)
    part = tmp_path / "models" / "diffusion_models" / "model.safetensors.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(data[:4])
    response = _Response(
        206, headers={"Content-Range": content_range}, chunks=[data[4:]]
    )
    _install_aiohttp(monkeypatch, _Session([response]))
    with pytest.raises(JobError, match="Content-Range|size"):
        _run(manager._download(job))
    assert part.read_bytes() == data[:4]
    assert not part.with_name("model.safetensors").exists()


def test_resume_falls_back_to_full_replacement_on_http_200(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accept_safetensors: None,
    public_dns: None,
) -> None:
    manager = _manager(tmp_path)
    data = b"new-complete-content"
    job = _job_for_bytes(manager, data)
    part = tmp_path / "models" / "diffusion_models" / "model.safetensors.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"old-partial")
    response = _Response(
        200,
        headers={"Content-Length": str(len(data))},
        chunks=[data],
    )
    session = _Session([response])
    _install_aiohttp(monkeypatch, session)
    _run(manager._download(job))
    assert part.with_name("model.safetensors").read_bytes() == data
    assert session.calls[0][1]["headers"]["Range"] == f"bytes={len(b'old-partial')}-"


def test_http_416_only_completes_an_already_full_verified_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accept_safetensors: None,
    public_dns: None,
) -> None:
    manager = _manager(tmp_path)
    data = b"complete"
    job = _job_for_bytes(manager, data)
    part = tmp_path / "models" / "diffusion_models" / "model.safetensors.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(data)
    _install_aiohttp(monkeypatch, _Session([_Response(416)]))
    _run(manager._download(job))
    assert part.with_name("model.safetensors").read_bytes() == data

    other = _job_for_bytes(manager, data, filename="incomplete.safetensors")
    other_part = part.with_name("incomplete.safetensors.part")
    other_part.write_bytes(data[:-1])
    _install_aiohttp(monkeypatch, _Session([_Response(416)]))
    with pytest.raises(JobError, match="HTTP 416"):
        _run(manager._download(other))
    assert other_part.exists()
    assert not other_part.with_name("incomplete.safetensors").exists()


def test_verified_final_recovery_removes_only_same_inode_partial_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accept_safetensors: None,
) -> None:
    manager = _manager(tmp_path)
    data = b"complete"
    job = _job_for_bytes(manager, data)
    final = tmp_path / "models" / "diffusion_models" / "model.safetensors"
    part = final.with_name("model.safetensors.part")
    final.parent.mkdir(parents=True)
    final.write_bytes(data)
    os.link(final, part)
    _install_aiohttp(monkeypatch, _Session([]))
    _run(manager._download(job))
    assert final.read_bytes() == data
    assert not part.exists()


def test_verified_final_recovery_preserves_separate_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accept_safetensors: None,
) -> None:
    manager = _manager(tmp_path)
    data = b"complete"
    job = _job_for_bytes(manager, data)
    final = tmp_path / "models" / "diffusion_models" / "model.safetensors"
    part = final.with_name("model.safetensors.part")
    final.parent.mkdir(parents=True)
    final.write_bytes(data)
    part.write_bytes(data)
    _install_aiohttp(monkeypatch, _Session([]))
    _run(manager._download(job))
    assert final.read_bytes() == data
    assert part.read_bytes() == data


def test_hash_mismatch_retains_partial_and_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accept_safetensors: None,
    public_dns: None,
) -> None:
    manager = _manager(tmp_path)
    expected = b"expected"
    received = b"tampered"
    assert len(expected) == len(received)
    job = _job_for_bytes(manager, expected)
    response = _Response(
        200,
        headers={"Content-Length": str(len(received))},
        chunks=[received],
    )
    _install_aiohttp(monkeypatch, _Session([response]))
    with pytest.raises(JobError, match="SHA256"):
        _run(manager._download(job))
    target = tmp_path / "models" / "diffusion_models" / "model.safetensors"
    assert not target.exists()
    assert target.with_name("model.safetensors.part").read_bytes() == received


def test_final_disk_rehash_detects_concurrent_partial_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accept_safetensors: None,
    public_dns: None,
) -> None:
    manager = _manager(tmp_path)
    data = b"expected-on-disk"
    job = _job_for_bytes(manager, data)
    part = tmp_path / "models" / "diffusion_models" / "model.safetensors.part"
    real_hash_descriptor = jobs_module._hash_descriptor

    def mutate_then_hash(descriptor: int):  # noqa: ANN202
        with part.open("r+b") as stream:
            stream.write(b"X")
            stream.flush()
            os.fsync(stream.fileno())
        return real_hash_descriptor(descriptor)

    monkeypatch.setattr(jobs_module, "_hash_descriptor", mutate_then_hash)

    response = _Response(
        200,
        headers={"Content-Length": str(len(data))},
        chunks=[data],
    )
    _install_aiohttp(monkeypatch, _Session([response]))
    with pytest.raises(JobError, match="final SHA256"):
        _run(manager._download(job))
    assert part.exists()
    assert not part.with_name("model.safetensors").exists()


def test_safetensors_validation_failure_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,
) -> None:
    manager = _manager(tmp_path)
    data = b"not-really-safetensors"
    job = _job_for_bytes(manager, data)
    monkeypatch.setattr(
        jobs_module,
        "validate_safetensors_file",
        lambda path: (_ for _ in ()).throw(SafeTensorsError("invalid safetensors")),
    )
    _install_aiohttp(
        monkeypatch,
        _Session(
            [
                _Response(
                    200,
                    headers={"Content-Length": str(len(data))},
                    chunks=[data],
                )
            ]
        ),
    )
    with pytest.raises(SafeTensorsError, match="invalid"):
        _run(manager._download(job))
    final = tmp_path / "models" / "diffusion_models" / "model.safetensors"
    assert not final.exists()
    assert final.with_name("model.safetensors.part").exists()


def test_symlinked_partial_and_final_are_rejected_without_touching_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    data = b"model"
    job = _job_for_bytes(manager, data)
    directory = tmp_path / "models" / "diffusion_models"
    directory.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"preserve")
    part = directory / "model.safetensors.part"
    part.symlink_to(outside)
    _install_aiohttp(monkeypatch, _Session([]))
    with pytest.raises(SecurityError, match="regular"):
        _run(manager._download(job))
    assert outside.read_bytes() == b"preserve"
    part.unlink()
    final = directory / "model.safetensors"
    final.symlink_to(outside)
    with pytest.raises(SecurityError, match="regular"):
        _run(manager._download(job))
    assert outside.read_bytes() == b"preserve"


def test_publication_is_atomic_and_never_overwrites_racing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accept_safetensors: None,
    public_dns: None,
) -> None:
    manager = _manager(tmp_path)
    data = b"downloaded"
    job = _job_for_bytes(manager, data)
    final = tmp_path / "models" / "diffusion_models" / "model.safetensors"
    racer = b"created-by-another-process"

    def racing_link(descriptor: int, directory_fd: int, destination_name: str) -> None:
        del descriptor
        destination_fd = os.open(
            destination_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o640,
            dir_fd=directory_fd,
        )
        try:
            os.write(destination_fd, racer)
        finally:
            os.close(destination_fd)
        raise FileExistsError(destination_name)

    monkeypatch.setattr(
        jobs_module, "_link_validated_descriptor_no_replace", racing_link
    )
    response = _Response(
        200,
        headers={"Content-Length": str(len(data))},
        chunks=[data],
    )
    _install_aiohttp(monkeypatch, _Session([response]))
    with pytest.raises((JobError, FileExistsError), match="destination|exist"):
        _run(manager._download(job))
    assert final.read_bytes() == racer


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="exercises the production Linux descriptor-bound linkat path",
)
def test_source_name_replacement_during_link_fails_without_publishing_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accept_safetensors: None,
    public_dns: None,
) -> None:
    manager = _manager(tmp_path)
    data = b"verified-model"
    replacement = b"tampered-model"
    assert len(data) == len(replacement)
    job = _job_for_bytes(manager, data)
    part = tmp_path / "models" / "diffusion_models" / "model.safetensors.part"
    final = part.with_name("model.safetensors")
    real_link = jobs_module._link_validated_descriptor_no_replace

    def racing_link(descriptor: int, directory_fd: int, destination_name: str) -> None:
        source_name = f"{destination_name}.part"
        replacement_name = f"{source_name}.replacement"
        replacement_fd = os.open(
            replacement_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o640,
            dir_fd=directory_fd,
        )
        try:
            os.write(replacement_fd, replacement)
        finally:
            os.close(replacement_fd)
        os.replace(
            replacement_name,
            source_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        real_link(descriptor, directory_fd, destination_name)

    monkeypatch.setattr(
        jobs_module, "_link_validated_descriptor_no_replace", racing_link
    )
    response = _Response(
        200,
        headers={"Content-Length": str(len(data))},
        chunks=[data],
    )
    _install_aiohttp(monkeypatch, _Session([response]))
    with pytest.raises(OSError):
        _run(manager._download(job))
    assert not final.exists()
    assert part.read_bytes() == replacement


def test_racing_replacement_after_link_is_preserved_and_job_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accept_safetensors: None,
    public_dns: None,
) -> None:
    manager = _manager(tmp_path)
    data = b"verified-model"
    racer = b"racing-file---"
    assert len(data) == len(racer)
    job = _job_for_bytes(manager, data)
    final = tmp_path / "models" / "diffusion_models" / "model.safetensors"
    real_link = jobs_module._link_validated_descriptor_no_replace

    def replace_after_link(
        descriptor: int, directory_fd: int, destination_name: str
    ) -> None:
        real_link(descriptor, directory_fd, destination_name)
        os.unlink(destination_name, dir_fd=directory_fd)
        racer_fd = os.open(
            destination_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o640,
            dir_fd=directory_fd,
        )
        try:
            os.write(racer_fd, racer)
        finally:
            os.close(racer_fd)

    monkeypatch.setattr(
        jobs_module, "_link_validated_descriptor_no_replace", replace_after_link
    )
    response = _Response(
        200,
        headers={"Content-Length": str(len(data))},
        chunks=[data],
    )
    _install_aiohttp(monkeypatch, _Session([response]))
    with pytest.raises(JobError, match="changed immediately"):
        _run(manager._download(job))
    assert final.read_bytes() == racer


def test_safetensors_validation_happens_before_final_name_is_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,
) -> None:
    manager = _manager(tmp_path)
    data = b"verified-model"
    job = _job_for_bytes(manager, data)
    final = tmp_path / "models" / "diffusion_models" / "model.safetensors"

    def validate_before_publish(path: Path) -> dict[str, int]:
        assert not final.exists()
        return {"size": Path(path).stat().st_size, "tensor_count": 1}

    monkeypatch.setattr(
        jobs_module, "validate_safetensors_file", validate_before_publish
    )
    response = _Response(
        200,
        headers={"Content-Length": str(len(data))},
        chunks=[data],
    )
    _install_aiohttp(monkeypatch, _Session([response]))
    _run(manager._download(job))
    assert final.read_bytes() == data


def test_published_inode_is_rehashed_after_validator_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,
) -> None:
    manager = _manager(tmp_path)
    data = b"verified-model"
    replacement = b"tampered-model"
    assert len(data) == len(replacement)
    job = _job_for_bytes(manager, data)
    part = tmp_path / "models" / "diffusion_models" / "model.safetensors.part"
    final = part.with_name("model.safetensors")

    def mutate_linked_inode(path: Path) -> dict[str, int]:
        part.write_bytes(replacement)
        return {"size": Path(path).stat().st_size, "tensor_count": 1}

    monkeypatch.setattr(jobs_module, "validate_safetensors_file", mutate_linked_inode)
    response = _Response(
        200,
        headers={"Content-Length": str(len(data))},
        chunks=[data],
    )
    _install_aiohttp(monkeypatch, _Session([response]))
    with pytest.raises(JobError, match="changed during validation"):
        _run(manager._download(job))
    assert not final.exists()
    assert part.read_bytes() == replacement


def test_service_restart_recovers_downloading_job_and_resumes_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accept_safetensors: None,
    public_dns: None,
) -> None:
    signer = TokenSigner(tmp_path / "state")
    original = _manager(tmp_path, signer=signer)
    data = b"restart-resume"
    token = _token(
        signer,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    [created] = _run(original.create_jobs([token], True, SUBJECT))
    original._jobs[created["id"]]["status"] = "downloading"
    original._jobs[created["id"]]["bytes_downloaded"] = 4
    original._persist()
    part = tmp_path / "models" / "diffusion_models" / "model.safetensors.part"
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(data[:4])
    original._instance_lease.release()

    recovered = _manager(tmp_path, signer=signer)
    [job] = recovered._jobs.values()
    assert job["status"] == "queued"
    assert "Recovered" in job["error"]
    response = _Response(
        206,
        headers={
            "Content-Range": f"bytes 4-{len(data) - 1}/{len(data)}",
            "Content-Length": str(len(data) - 4),
        },
        chunks=[data[4:]],
    )
    _install_aiohttp(monkeypatch, _Session([response]))
    _run(recovered._download(job))
    assert part.with_name("model.safetensors").read_bytes() == data
    assert job["status"] == "completed"


def test_authorization_is_sent_only_to_provider_origin_never_redirect_cdn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accept_safetensors: None,
    public_dns: None,
) -> None:
    manager = _manager(tmp_path)
    data = b"download"
    job = _job_for_bytes(manager, data)
    monkeypatch.setenv("HF_TOKEN", "top-secret")
    redirect = _Response(
        302,
        headers={"Location": "https://cdn-lfs.huggingface.co/object"},
    )
    body = _Response(
        200,
        headers={"Content-Length": str(len(data))},
        chunks=[data],
    )
    session = _Session([redirect, body])
    _install_aiohttp(monkeypatch, session)
    _run(manager._download(job))
    origin_headers = session.calls[0][1]["headers"]
    cdn_headers = session.calls[1][1]["headers"]
    assert origin_headers["Authorization"] == "Bearer top-secret"
    assert "Authorization" not in cdn_headers
    assert redirect.released and body.released


def test_only_one_worker_task_runs_jobs_serially(tmp_path: Path) -> None:
    class RecordingManager(JobManager):
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            super().__init__(*args, **kwargs)
            self.active = 0
            self.maximum_active = 0
            self.order: list[str] = []
            self.finished = asyncio.Event()

        async def _run_job(self, job: dict[str, Any]) -> None:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            self.order.append(job["id"])
            await asyncio.sleep(0)
            job["status"] = "completed"
            self.active -= 1
            if len(self.order) == 2:
                self.finished.set()

    async def scenario() -> None:
        signer = TokenSigner(tmp_path / "state")
        (tmp_path / "models").mkdir()
        manager = RecordingManager(
            tmp_path / "models", tmp_path / "state", signer=signer, reserve_bytes=0
        )
        await manager.create_jobs(
            [
                _token(signer, filename="one.safetensors", size=1, sha256="1" * 64),
                _token(signer, filename="two.safetensors", size=1, sha256="2" * 64),
            ],
            True,
            SUBJECT,
        )
        first_task = manager._worker_task
        await manager.ensure_started()
        assert manager._worker_task is first_task
        await asyncio.wait_for(manager.finished.wait(), timeout=1)
        assert manager.maximum_active == 1
        assert len(manager.order) == 2
        assert first_task is not None
        first_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_task

    _run(scenario())


def test_v1_state_is_migrated_to_v2_with_additive_fields(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    job = _job_for_bytes(manager, b"migrate")
    legacy = {
        key: value
        for key, value in job.items()
        if key
        not in {"phase", "error_code", "attempt", "max_attempts", "owner_binding"}
    }
    failed_legacy = dict(legacy)
    failed_legacy.update(
        {
            "id": "failed-job",
            "status": "failed",
            "error": "[Errno 5] failed: '/private/legacy/model.safetensors'",
            "created_at": jobs_module._now(),
            "updated_at": jobs_module._now(),
        }
    )
    manager.jobs_path.write_text(
        json.dumps({"version": 1, "jobs": [legacy, failed_legacy]}),
        encoding="utf-8",
    )
    signer = manager.signer
    manager._instance_lease.release()

    recovered = _manager(tmp_path, signer=signer)
    migrated = recovered._jobs["job"]
    assert migrated["phase"] == "queued"
    assert migrated["error_code"] is None
    assert migrated["attempt"] == 0
    assert migrated["max_attempts"] == 5
    failed = recovered._jobs["failed-job"]
    assert failed["error"] == "Previous download failed"
    assert failed["error_code"] == "legacy_failure"
    assert "/private/legacy" not in recovered.jobs_path.read_text("utf-8")
    assert json.loads(recovered.jobs_path.read_text("utf-8"))["version"] == 2
    assert migrated["owner_binding"] is None
    with pytest.raises(KeyError, match="not found"):
        recovered.get_job("job", SUBJECT)


def test_corrupt_state_is_quarantined_and_service_starts_empty(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    state = tmp_path / "state"
    signer = TokenSigner(state)
    (state / "jobs.json").write_text("{broken", encoding="utf-8")

    manager = JobManager(models, state, signer=signer, reserve_bytes=0)
    assert manager.list_jobs(SUBJECT) == []
    assert manager.health()["quarantined_state"] is True
    assert json.loads((state / "jobs.json").read_text("utf-8")) == {
        "jobs": [],
        "version": 2,
    }
    assert len(list(state.glob("jobs.corrupt-*.json"))) == 1


def test_history_retention_caps_terminal_jobs_and_keeps_active(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    now = jobs_module._now()
    for index in range(1_005):
        job = _job_for_bytes(manager, b"history", filename=f"model-{index}.safetensors")
        job.update(
            {
                "id": f"job-{index:04d}",
                "status": "completed",
                "phase": "completed",
                "created_at": now + index,
                "updated_at": now + index,
                "completed_at": now + index,
                "attempt": 1,
                "max_attempts": 5,
                "error_code": None,
            }
        )
        manager._jobs[job["id"]] = job
    active = _job_for_bytes(manager, b"active", filename="active.safetensors")
    active.update(
        {
            "id": "active",
            "phase": "queued",
            "created_at": 1.0,
            "updated_at": 1.0,
            "attempt": 0,
            "max_attempts": 5,
            "error_code": None,
        }
    )
    manager._jobs[active["id"]] = active
    expired = _job_for_bytes(manager, b"expired", filename="expired.safetensors")
    expired.update(
        {
            "id": "expired",
            "status": "completed",
            "phase": "completed",
            "created_at": now + 10_000,
            "updated_at": now + 10_000,
            "completed_at": now - (91 * 24 * 60 * 60),
            "attempt": 1,
            "max_attempts": 5,
            "error_code": None,
        }
    )
    manager._jobs[expired["id"]] = expired
    manager._persist()
    assert len(manager._jobs) == 1_001
    assert "active" in manager._jobs
    assert "expired" not in manager._jobs
    assert "job-0000" not in manager._jobs


def test_history_cursor_is_stable_and_opaque(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    for index in range(3):
        job = _job_for_bytes(manager, b"page", filename=f"page-{index}.safetensors")
        job.update(
            {
                "id": f"page-{index}",
                "phase": "queued",
                "created_at": float(index),
                "attempt": 0,
                "max_attempts": 5,
                "error_code": None,
            }
        )
        manager._jobs[job["id"]] = job
    first = manager.list_job_history(SUBJECT, limit=2)
    second = manager.list_job_history(SUBJECT, cursor=first["next_cursor"], limit=2)
    assert [job["id"] for job in first["jobs"]] == ["page-2", "page-1"]
    assert [job["id"] for job in second["jobs"]] == ["page-0"]
    assert second["next_cursor"] is None


def test_retryable_failure_succeeds_on_second_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    job = _job_for_bytes(manager, b"retry")
    job.update(
        {
            "phase": "queued",
            "attempt": 0,
            "max_attempts": 5,
            "error_code": None,
        }
    )
    manager._jobs[job["id"]] = job
    calls = 0

    async def download(target: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableDownloadError(
                "provider is busy",
                error_code="upstream_server_error",
                retry_after=0,
            )
        target["status"] = "completed"
        target["phase"] = "completed"
        target["completed_at"] = jobs_module._now()

    monkeypatch.setattr(manager, "_download", download)
    _run(manager._run_job(job))
    assert calls == 2
    assert job["status"] == "completed"
    assert job["attempt"] == 2
    assert job["error_code"] is None


def test_retry_budget_is_limited_to_five_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    job = _job_for_bytes(manager, b"retry-limit")
    job.update(
        {
            "phase": "queued",
            "attempt": 0,
            "max_attempts": 5,
            "error_code": None,
        }
    )
    manager._jobs[job["id"]] = job
    calls = 0

    async def download(target: dict[str, Any]) -> None:
        nonlocal calls
        del target
        calls += 1
        raise RetryableDownloadError(
            "rate limited", error_code="upstream_rate_limited", retry_after=0
        )

    monkeypatch.setattr(manager, "_download", download)
    _run(manager._run_job(job))
    assert calls == 5
    assert job["status"] == "failed"
    assert job["phase"] == "failed"
    assert job["attempt"] == 5
    assert job["error_code"] == "upstream_rate_limited"


@pytest.mark.parametrize(
    ("status", "error_code"),
    [(429, "upstream_rate_limited"), (503, "upstream_server_error")],
)
def test_retryable_http_status_honors_retry_after(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,
    status: int,
    error_code: str,
) -> None:
    manager = _manager(tmp_path)
    job = _job_for_bytes(manager, b"provider-busy")
    response = _Response(status, headers={"Retry-After": "17"})
    _install_aiohttp(monkeypatch, _Session([response]))
    with pytest.raises(RetryableDownloadError) as caught:
        _run(manager._download(job))
    assert caught.value.error_code == error_code
    assert caught.value.retry_after == 17
    assert response.released is True


def test_path_bearing_oserror_is_redacted_from_persisted_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    job = _job_for_bytes(manager, b"redact")
    job.update(
        {
            "phase": "queued",
            "attempt": 0,
            "max_attempts": 5,
            "error_code": None,
        }
    )
    manager._jobs[job["id"]] = job

    async def fail(target: dict[str, Any]) -> None:
        del target
        raise OSError(5, "private path failed", "/private/secret/model.safetensors")

    monkeypatch.setattr(manager, "_download", fail)
    _run(manager._run_job(job))
    persisted = manager.get_job(job["id"], SUBJECT)
    assert persisted["error"] == "filesystem operation failed"
    assert persisted["error_code"] == "filesystem_error"
    assert "/private/secret" not in manager.jobs_path.read_text("utf-8")


def test_cancellation_after_hash_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accept_safetensors: None,
    public_dns: None,
) -> None:
    manager = _manager(tmp_path)
    data = b"cancel-before-publication"
    job = _job_for_bytes(manager, data)
    job.update(
        {
            "phase": "downloading",
            "attempt": 1,
            "max_attempts": 5,
            "error_code": None,
        }
    )
    real_hash = jobs_module._hash_descriptor

    def hash_then_cancel(descriptor: int):  # noqa: ANN202
        result = real_hash(descriptor)
        job["cancel_requested"] = True
        return result

    monkeypatch.setattr(jobs_module, "_hash_descriptor", hash_then_cancel)
    _install_aiohttp(
        monkeypatch,
        _Session(
            [
                _Response(
                    200,
                    headers={"Content-Length": str(len(data))},
                    chunks=[data],
                )
            ]
        ),
    )
    with pytest.raises(DownloadCancelled):
        _run(manager._download(job))
    final = tmp_path / "models" / "diffusion_models" / "model.safetensors"
    assert not final.exists()
    assert final.with_name("model.safetensors.part").exists()


def test_discard_partial_is_locked_and_terminal_only(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    job = _job_for_bytes(manager, b"discard")
    job.update(
        {
            "status": "failed",
            "phase": "failed",
            "attempt": 1,
            "max_attempts": 5,
            "error_code": "network_error",
        }
    )
    manager._jobs[job["id"]] = job
    _, part = jobs_module.resolve_model_paths(
        manager.models_root, job["directory"], job["filename"]
    )
    part.write_bytes(b"partial")
    discarded = _run(manager.discard_partial(job["id"], SUBJECT))
    assert discarded["bytes_downloaded"] == 0
    assert not part.exists()
    job["status"] = "completed"
    job["phase"] = "completed"
    with pytest.raises(JobError, match="terminal"):
        _run(manager.discard_partial(job["id"], SUBJECT))


def test_start_health_and_controlled_stop_release_singleton(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    async def scenario() -> None:
        await manager.ensure_started()
        assert manager.health()["worker_running"] is True
        await manager.stop()

    _run(scenario())
    health = manager.health()
    assert health["state"] == "stopped"
    assert health["instance_lock_acquired"] is False
