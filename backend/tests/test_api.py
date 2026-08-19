from __future__ import annotations

import asyncio
import hashlib
import json
import types
from pathlib import Path
from typing import Any

import pytest

from backend import api as api_module
from backend.api import (
    API_PREFIX,
    API_VERSION,
    EXTENSION_VERSION,
    MAX_BODY_BYTES,
    MAX_JOBS_PER_REQUEST,
    MAX_MODELS_PER_INSPECTION,
    ServerModelDownloaderAPI,
    register_routes,
)
from backend.jobs import JobError
from backend.metadata import MetadataError
from backend.security import ALLOWED_DIRECTORIES, SAFE_EXTENSIONS, TokenSigner
from backend.settings import RuntimeSettings

TEST_IDENTITY = "test-user@example.com"
TEST_SUBJECT = f"trusted-proxy:{TEST_IDENTITY}"
TEST_ORIGIN = "https://comfy.example.com"


def _run(coro):  # noqa: ANN001, ANN202
    return asyncio.run(coro)


def _response_data(response: Any) -> Any:
    """Read both aiohttp.web.Response and the dependency-free fallback."""

    if hasattr(response, "data"):
        return response.data
    return json.loads(response.text)


class _BodyStream:
    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks

    async def iter_chunked(self, size: int):  # noqa: ANN201
        for chunk in self.chunks:
            yield chunk


class _Request:
    def __init__(
        self,
        body: object | bytes | None = None,
        *,
        headers: dict[str, str] | None = None,
        content_length: object = "auto",
        match_info: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
        query: dict[str, str] | None = None,
        authenticated: bool = True,
    ) -> None:
        if isinstance(body, bytes):
            raw = body
        elif body is None:
            raw = b""
        else:
            raw = json.dumps(body).encode()
        identity = {"X-Forwarded-User": TEST_IDENTITY} if authenticated else {}
        self.headers = {
            "Content-Type": "application/json",
            **identity,
            **(headers or {}),
        }
        self.content_length = len(raw) if content_length == "auto" else content_length
        self.content = _BodyStream(chunks if chunks is not None else [raw])
        self.match_info = match_info or {}
        self.query = query or {}
        self.remote = "127.0.0.1"


def _authority_headers(signer: TokenSigner) -> dict[str, str]:
    csrf_token, _ = signer.issue_csrf(TEST_SUBJECT, TEST_ORIGIN)
    return {
        "X-SMD-CSRF": csrf_token,
        "Origin": TEST_ORIGIN,
        "Sec-Fetch-Site": "same-origin",
    }


class _Manager:
    def __init__(self) -> None:
        self.started = 0
        self.created: list[tuple[list[str], bool, str]] = []
        self.cancelled: list[str] = []
        self.accessed_subjects: list[str] = []
        self.jobs: dict[str, dict[str, Any]] = {
            "job-1": {
                "id": "job-1",
                "name": "ae.safetensors",
                "filename": "ae.safetensors",
                "directory": "vae",
                "status": "queued",
                "bytes_downloaded": 0,
                "size": 1024,
                "progress": 0.0,
                "error": None,
                "sha256": "a" * 64,
                "canonical_url": "must-not-be-exposed",
                "download_token": "must-not-be-exposed",
                "authorization": "must-not-be-exposed",
            }
        }

    async def ensure_started(self) -> None:
        self.started += 1

    async def create_jobs(self, tokens: list[str], confirmed: bool, subject: str):  # noqa: ANN201
        self.created.append((tokens, confirmed, subject))
        return [self.jobs["job-1"]]

    async def list_jobs(self, subject: str):  # noqa: ANN201
        self.accessed_subjects.append(subject)
        return list(self.jobs.values())

    def list_job_history(  # noqa: ANN201
        self,
        subject: str,
        *,
        cursor=None,
        limit=50,  # noqa: ANN001
    ):
        self.accessed_subjects.append(subject)
        assert cursor in (None, "next")
        assert 1 <= limit <= 100
        return {"jobs": list(self.jobs.values())[:limit], "next_cursor": None}

    async def get_job(self, job_id: str, subject: str):  # noqa: ANN201
        self.accessed_subjects.append(subject)
        return self.jobs.get(job_id)

    async def cancel_job(self, job_id: str, subject: str):  # noqa: ANN201
        self.accessed_subjects.append(subject)
        if job_id not in self.jobs:
            return None
        self.cancelled.append(job_id)
        self.jobs[job_id]["status"] = "cancelled"
        return self.jobs[job_id]

    async def discard_partial(self, job_id: str, subject: str):  # noqa: ANN201
        self.accessed_subjects.append(subject)
        return self.jobs.get(job_id)

    def health(self):  # noqa: ANN201
        return {"status": "ok", "state": "running", "worker_running": True}

    async def stop(self) -> None:
        return None


def _make_api(tmp_path: Path, manager: _Manager | None = None):  # noqa: ANN201
    (tmp_path / "models").mkdir(exist_ok=True)
    signer = TokenSigner(tmp_path / "state")
    settings = RuntimeSettings.from_env(
        {
            "SMD_ENABLED": "true",
            "SMD_PUBLIC_ORIGIN": TEST_ORIGIN,
            "SMD_AUTH_MODE": "trusted-proxy",
            "SMD_MODELS_ROOT": str(tmp_path / "models"),
            "SMD_STATE_DIR": str(tmp_path / "state"),
        }
    )
    instance = ServerModelDownloaderAPI(
        settings=settings,
        signer=signer,
        job_manager=manager,
    )
    return instance, signer


def test_session_is_no_store_and_discloses_only_capabilities(tmp_path: Path) -> None:
    api, _ = _make_api(tmp_path)
    response = _run(api.session(_Request()))
    assert response.status == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.content_type == "application/json"
    data = _response_data(response)
    assert data["api_version"] == API_VERSION
    assert data["extension_version"] == EXTENSION_VERSION
    assert data["csrf_token"]
    assert data["csrf_expires_at"] > 0
    assert data["allowed_directories"] == list(ALLOWED_DIRECTORIES)
    assert data["safe_extensions"] == list(SAFE_EXTENSIONS)
    assert data["identity"] == {
        "email": TEST_IDENTITY,
        "auth_mode": "trusted-proxy",
    }
    assert data["capabilities"]["max_models_per_scan"] == 50


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {
            "X-SMD-CSRF": "wrong",
            "Origin": TEST_ORIGIN,
        },
        {
            "X-SMD-CSRF": "placeholder",
            "Origin": "https://evil.example",
        },
    ],
)
def test_mutations_require_csrf_and_same_origin(
    tmp_path: Path, headers: dict[str, str]
) -> None:
    api, signer = _make_api(tmp_path, _Manager())
    if headers.get("X-SMD-CSRF") == "placeholder":
        headers["X-SMD-CSRF"] = _authority_headers(signer)["X-SMD-CSRF"]
    response = _run(
        api.create_jobs(
            _Request(
                {"download_tokens": ["signed"], "license_confirmed": True},
                headers=headers,
            )
        )
    )
    assert response.status == 403
    assert _response_data(response)["code"] == "forbidden"
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize(
    ("case_request", "status", "code"),
    [
        (
            _Request({}, headers={"Content-Type": "text/plain"}),
            415,
            "unsupported_media_type",
        ),
        (_Request(b"", content_length=0), 400, "invalid_json"),
        (_Request(b"{"), 400, "invalid_json"),
        (_Request({}, content_length="bad"), 400, "invalid_request"),
        (_Request({}, content_length=-1), 400, "invalid_request"),
        (_Request({}, content_length=MAX_BODY_BYTES + 1), 413, "body_too_large"),
        (
            _Request(
                b"",
                content_length=None,
                chunks=[b"x" * MAX_BODY_BYTES, b"x"],
            ),
            413,
            "body_too_large",
        ),
    ],
)
def test_json_body_parser_is_bounded_and_strict(
    tmp_path: Path, case_request: _Request, status: int, code: str
) -> None:
    api, signer = _make_api(tmp_path, _Manager())
    case_request.headers.update(_authority_headers(signer))
    response = _run(api.create_jobs(case_request))
    assert response.status == status
    assert _response_data(response)["code"] == code


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ([], "invalid_request"),
        ({}, "invalid_request"),
        ({"download_tokens": []}, "invalid_request"),
        ({"download_tokens": [123], "license_confirmed": True}, "invalid_request"),
        (
            {"download_tokens": ["signed"], "license_confirmed": False},
            "license_required",
        ),
        ({"download_tokens": ["signed"]}, "license_required"),
        (
            {
                "download_tokens": ["signed"] * (MAX_JOBS_PER_REQUEST + 1),
                "license_confirmed": True,
            },
            "too_many_jobs",
        ),
    ],
)
def test_create_jobs_rejects_invalid_schema_and_requires_license_confirmation(
    tmp_path: Path, body: object, code: str
) -> None:
    api, signer = _make_api(tmp_path, _Manager())
    response = _run(api.create_jobs(_Request(body, headers=_authority_headers(signer))))
    assert response.status == 400
    assert _response_data(response)["code"] == code


def test_create_list_get_and_cancel_jobs_redact_internal_secrets(
    tmp_path: Path,
) -> None:
    manager = _Manager()
    api, signer = _make_api(tmp_path, manager)
    response = _run(
        api.create_jobs(
            _Request(
                {"download_tokens": ["signed"], "license_confirmed": True},
                headers=_authority_headers(signer),
            )
        )
    )
    assert response.status == 202
    assert manager.created == [(["signed"], True, TEST_SUBJECT)]
    job = _response_data(response)["jobs"][0]
    assert job["id"] == "job-1"
    assert "canonical_url" not in job
    assert "download_token" not in job
    assert "authorization" not in job

    listed = _run(api.list_jobs(_Request()))
    assert listed.status == 200
    assert _response_data(listed)["jobs"][0]["id"] == "job-1"
    fetched = _run(api.get_job(_Request(match_info={"job_id": "job-1"})))
    assert fetched.status == 200
    assert _response_data(fetched)["job"]["id"] == "job-1"

    cancelled = _run(
        api.cancel_job(
            _Request(
                {},
                headers=_authority_headers(signer),
                match_info={"job_id": "job-1"},
            )
        )
    )
    assert cancelled.status == 202
    assert _response_data(cancelled)["job"]["status"] == "cancelled"
    assert manager.cancelled == ["job-1"]
    # Every manager access rechecks the supervised worker.
    assert manager.started == 4


@pytest.mark.parametrize("job_id", ["", "../x", "/tmp/x", "x y", "☃", "x" * 129])
def test_job_id_is_validated_before_manager_lookup(tmp_path: Path, job_id: str) -> None:
    api, _ = _make_api(tmp_path, _Manager())
    response = _run(api.get_job(_Request(match_info={"job_id": job_id})))
    assert response.status == 400
    assert _response_data(response)["code"] == "invalid_job_id"


def test_unknown_job_returns_404_for_get_and_cancel(tmp_path: Path) -> None:
    api, signer = _make_api(tmp_path, _Manager())
    fetched = _run(api.get_job(_Request(match_info={"job_id": "missing"})))
    assert fetched.status == 404
    assert _response_data(fetched)["code"] == "not_found"
    cancelled = _run(
        api.cancel_job(
            _Request(
                {},
                headers=_authority_headers(signer),
                match_info={"job_id": "missing"},
            )
        )
    )
    assert cancelled.status == 404
    assert _response_data(cancelled)["code"] == "not_found"


class _Inspector:
    def __init__(
        self,
        candidates: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
    ):
        self.candidates = candidates or []
        self.error = error
        self.calls: list[object] = []

    async def inspect(self, session, model, *, subject):  # noqa: ANN001, ANN201
        assert subject == TEST_SUBJECT
        self.calls.append(model)
        if self.error:
            raise self.error
        return self.candidates


class _ClientSession:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):  # noqa: ANN204
        return self

    async def __aexit__(self, exc_type, exc, traceback):  # noqa: ANN001, ANN204
        return False


def _enable_fake_aiohttp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api_module,
        "_aiohttp",
        types.SimpleNamespace(
            ClientTimeout=lambda **kwargs: kwargs,
            ClientSession=_ClientSession,
        ),
    )


def _candidate(token: str = "signed", *, sha256: str | None = None) -> dict[str, Any]:
    return {
        "provider": "huggingface",
        "requested_name": "ae.safetensors",
        "filename": "ae.safetensors",
        "source_filename": "ae.safetensors",
        "directory": "vae",
        "relative_path": "vae/ae.safetensors",
        "canonical_url": "https://huggingface.co/org/repo/resolve/"
        + "a" * 40
        + "/ae.safetensors",
        "revision": "a" * 40,
        "size": 4,
        "sha256": sha256 or hashlib.sha256(b"data").hexdigest(),
        "license": "apache-2.0",
        "license_url": "https://huggingface.co/org/repo/blob/" + "a" * 40 + "/LICENSE",
        "metadata": {},
        "expires_at": 9999999999,
        "download_token": token,
    }


def test_inspect_publishes_eligible_missing_model_and_detects_installed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_fake_aiohttp(monkeypatch)
    monkeypatch.setattr(
        api_module,
        "validate_safetensors_file",
        lambda path: {"size": Path(path).stat().st_size, "tensor_count": 1},
    )
    api, signer = _make_api(tmp_path)
    inspector = _Inspector([_candidate()])
    api.runtime._inspector = inspector
    model = {
        "name": "ae.safetensors",
        "url": "https://huggingface.co/org/repo/resolve/main/ae.safetensors",
        "directory": "auto",
    }

    missing = _run(
        api.inspect_models(
            _Request({"models": [model]}, headers=_authority_headers(signer))
        )
    )
    assert missing.status == 200
    [published] = _response_data(missing)["models"]
    assert published["eligible"] is True
    assert published["installed"] is False
    assert published["download_token"] == "signed"
    assert published["source"] == "Hugging Face"
    assert published["url"] == model["url"]
    assert len(published["id"]) == 24

    target = tmp_path / "models" / "vae" / "ae.safetensors"
    target.write_bytes(b"data")
    installed = _run(
        api.inspect_models(
            _Request({"models": [model]}, headers=_authority_headers(signer))
        )
    )
    [published] = _response_data(installed)["models"]
    assert published["installed"] is True
    assert published["eligible"] is False
    assert published["download_token"] is None


def test_inspect_rejects_same_size_existing_file_with_wrong_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_fake_aiohttp(monkeypatch)
    api, signer = _make_api(tmp_path)
    api.runtime._inspector = _Inspector([_candidate()])
    target = tmp_path / "models" / "vae" / "ae.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"nope")
    response = _run(
        api.inspect_models(
            _Request(
                {
                    "models": [
                        {
                            "name": "ae.safetensors",
                            "url": "https://huggingface.co/org/repo/resolve/main/ae.safetensors",
                            "directory": "vae",
                        }
                    ]
                },
                headers=_authority_headers(signer),
            )
        )
    )
    [published] = _response_data(response)["models"]
    assert published["installed"] is False
    assert published["eligible"] is False
    assert published["download_token"] is None
    assert "SHA256" in published["reason"]


def test_inspect_rejects_conflicting_existing_file_and_per_model_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_fake_aiohttp(monkeypatch)
    api, signer = _make_api(tmp_path)
    (tmp_path / "models" / "vae").mkdir(parents=True)
    (tmp_path / "models" / "vae" / "ae.safetensors").write_bytes(b"wrong-size")
    api.runtime._inspector = _Inspector([_candidate()])
    request_model = {
        "name": "ae.safetensors",
        "url": "https://huggingface.co/org/repo/resolve/main/ae.safetensors",
    }
    response = _run(
        api.inspect_models(
            _Request({"models": [request_model]}, headers=_authority_headers(signer))
        )
    )
    [published] = _response_data(response)["models"]
    assert published["eligible"] is False
    assert published["installed"] is False
    assert published["download_token"] is None
    assert "different size" in published["reason"]

    api.runtime._inspector = _Inspector(error=MetadataError("no verified SHA256"))
    response = _run(
        api.inspect_models(
            _Request({"models": [request_model]}, headers=_authority_headers(signer))
        )
    )
    [failed] = _response_data(response)["models"]
    assert failed["eligible"] is False
    assert failed["download_token"] is None
    assert failed["reason"] == "no verified SHA256"


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ([], "invalid_request"),
        ({}, "invalid_request"),
        ({"models": {}}, "invalid_request"),
        ({"models": [{}] * (MAX_MODELS_PER_INSPECTION + 1)}, "too_many_models"),
    ],
)
def test_inspect_request_schema_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body: object,
    code: str,
) -> None:
    _enable_fake_aiohttp(monkeypatch)
    api, signer = _make_api(tmp_path)
    response = _run(
        api.inspect_models(_Request(body, headers=_authority_headers(signer)))
    )
    assert response.status == 400
    assert _response_data(response)["code"] == code


class _Routes:
    def __init__(self) -> None:
        self.registered: list[tuple[str, str, Any]] = []

    def _decorator(self, method: str, path: str):  # noqa: ANN202
        def register(handler):  # noqa: ANN001, ANN202
            self.registered.append((method, path, handler))
            return handler

        return register

    def get(self, path: str):  # noqa: ANN201
        return self._decorator("GET", path)

    def post(self, path: str):  # noqa: ANN201
        return self._decorator("POST", path)

    def delete(self, path: str):  # noqa: ANN201
        return self._decorator("DELETE", path)


def test_register_routes_uses_existing_comfyui_http_server_only(tmp_path: Path) -> None:
    routes = _Routes()
    server = types.SimpleNamespace(
        routes=routes,
        app=types.SimpleNamespace(on_startup=[], on_cleanup=[]),
    )
    (tmp_path / "models").mkdir()
    settings = RuntimeSettings.from_env(
        {
            "SMD_ENABLED": "true",
            "SMD_PUBLIC_ORIGIN": TEST_ORIGIN,
            "SMD_AUTH_MODE": "trusted-proxy",
            "SMD_MODELS_ROOT": str(tmp_path / "models"),
            "SMD_STATE_DIR": str(tmp_path / "state"),
        }
    )
    registered = register_routes(
        server,
        settings=settings,
        signer=TokenSigner(tmp_path / "state"),
        job_manager=_Manager(),
    )
    assert isinstance(registered, ServerModelDownloaderAPI)
    assert [(method, path) for method, path, _ in routes.registered] == [
        ("GET", f"{API_PREFIX}/health"),
        ("GET", f"{API_PREFIX}/session"),
        ("POST", f"{API_PREFIX}/inspect"),
        ("POST", f"{API_PREFIX}/jobs"),
        ("GET", f"{API_PREFIX}/jobs"),
        ("GET", f"{API_PREFIX}/jobs/{{job_id}}"),
        ("POST", f"{API_PREFIX}/jobs/{{job_id}}/cancel"),
        ("DELETE", f"{API_PREFIX}/jobs/{{job_id}}/partial"),
    ]
    assert len(server.app.on_startup) == 1
    assert len(server.app.on_cleanup) == 1


def test_every_route_requires_authentication_and_errors_have_request_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_aiohttp(monkeypatch)
    api, _ = _make_api(tmp_path, _Manager())
    cases = [
        api.session(_Request(authenticated=False)),
        api.health(_Request(authenticated=False)),
        api.inspect_models(_Request({"models": []}, authenticated=False)),
        api.create_jobs(
            _Request(
                {"download_tokens": ["token"], "license_confirmed": True},
                authenticated=False,
            )
        ),
        api.list_jobs(_Request(authenticated=False)),
        api.get_job(_Request(match_info={"job_id": "job-1"}, authenticated=False)),
        api.cancel_job(
            _Request({}, match_info={"job_id": "job-1"}, authenticated=False)
        ),
        api.discard_partial(
            _Request(match_info={"job_id": "job-1"}, authenticated=False)
        ),
    ]
    for response in [_run(case) for case in cases]:
        data = _response_data(response)
        assert response.status == 401
        assert data["code"] == "authentication_required"
        assert len(data["request_id"]) == 32
        assert response.headers["X-Request-ID"] == data["request_id"]


def test_csrf_token_is_bound_to_identity_and_exact_origin(tmp_path: Path) -> None:
    api, signer = _make_api(tmp_path, _Manager())
    headers = _authority_headers(signer)
    headers["X-Forwarded-User"] = "different@example.com"
    response = _run(
        api.create_jobs(
            _Request(
                {"download_tokens": ["token"], "license_confirmed": True},
                headers=headers,
            )
        )
    )
    assert response.status == 403
    assert _response_data(response)["code"] == "forbidden"


def test_history_pagination_health_and_partial_cleanup(tmp_path: Path) -> None:
    manager = _Manager()
    api, signer = _make_api(tmp_path, manager)

    health = _run(api.health(_Request()))
    assert health.status == 200
    assert _response_data(health)["status"] == "ok"

    history = _run(api.list_jobs(_Request(query={"limit": "1"})))
    assert history.status == 200
    assert len(_response_data(history)["jobs"]) == 1
    assert _response_data(history)["next_cursor"] is None

    discarded = _run(
        api.discard_partial(
            _Request(
                headers=_authority_headers(signer),
                match_info={"job_id": "job-1"},
            )
        )
    )
    assert discarded.status == 200
    assert _response_data(discarded)["job"]["id"] == "job-1"
    assert manager.accessed_subjects == [TEST_SUBJECT, TEST_SUBJECT]


def test_worker_startup_failure_degrades_only_the_downloader(tmp_path: Path) -> None:
    class BrokenManager(_Manager):
        async def ensure_started(self) -> None:
            raise JobError("instance lease unavailable")

    api, _ = _make_api(tmp_path, BrokenManager())
    _run(api.runtime.startup())
    health = _run(api.health(_Request()))
    assert health.status == 503
    assert _response_data(health)["status"] == "degraded"
    session = _run(api.session(_Request()))
    assert session.status == 503
    assert _response_data(session)["code"] == "service_degraded"


def test_late_worker_degradation_makes_session_unavailable(tmp_path: Path) -> None:
    manager = _Manager()
    api, _ = _make_api(tmp_path, manager)
    manager.health = lambda: {"status": "degraded", "reason": "/private/state"}

    session = _run(api.session(_Request()))
    assert session.status == 503
    assert _response_data(session)["code"] == "service_degraded"
    assert "/private/state" not in session.text


def test_unexpected_api_errors_do_not_log_paths_or_tracebacks(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    class BrokenManager(_Manager):
        async def get_job(self, job_id: str, subject: str):  # noqa: ANN201
            del job_id, subject
            raise FileNotFoundError("/private/secret/model.safetensors")

    api, _ = _make_api(tmp_path, BrokenManager())
    with caplog.at_level("ERROR"):
        response = _run(api.get_job(_Request(match_info={"job_id": "job-1"})))

    assert response.status == 500
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "FileNotFoundError" in log_text
    assert "/private/secret" not in log_text
    assert all(record.exc_info is None for record in caplog.records)


def test_inspection_rate_limit_and_global_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_aiohttp(monkeypatch)
    api, signer = _make_api(tmp_path)
    active = 0
    maximum_active = 0

    class ConcurrentInspector:
        async def inspect(self, session, model, *, subject):  # noqa: ANN001, ANN201
            nonlocal active, maximum_active
            assert subject == TEST_SUBJECT
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return []

    api.runtime._inspector = ConcurrentInspector()
    models = [
        {
            "name": f"model-{index}.safetensors",
            "url": (
                "https://huggingface.co/org/repo/resolve/main/"
                f"model-{index}.safetensors"
            ),
        }
        for index in range(8)
    ]
    response = _run(
        api.inspect_models(
            _Request({"models": models}, headers=_authority_headers(signer))
        )
    )
    assert response.status == 200
    assert maximum_active == api_module.INSPECTION_CONCURRENCY

    api._inspection_rate_limiter = api_module._InspectionRateLimiter(limit=1)
    first = _run(
        api.inspect_models(_Request({"models": []}, headers=_authority_headers(signer)))
    )
    second = _run(
        api.inspect_models(_Request({"models": []}, headers=_authority_headers(signer)))
    )
    assert first.status == 200
    assert second.status == 429
    assert _response_data(second)["code"] == "rate_limited"


def test_inspection_deadline_cancels_provider_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_aiohttp(monkeypatch)
    monkeypatch.setattr(api_module, "INSPECTION_DEADLINE_SECONDS", 0.01)
    api, signer = _make_api(tmp_path)
    cancelled = False

    class SlowInspector:
        async def inspect(self, session, model, *, subject):  # noqa: ANN001, ANN201
            nonlocal cancelled
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                cancelled = True
                raise
            return []

    api.runtime._inspector = SlowInspector()
    response = _run(
        api.inspect_models(
            _Request(
                {
                    "models": [
                        {
                            "name": "model.safetensors",
                            "url": (
                                "https://huggingface.co/org/repo/resolve/main/"
                                "model.safetensors"
                            ),
                        }
                    ]
                },
                headers=_authority_headers(signer),
            )
        )
    )
    assert response.status == 504
    assert _response_data(response)["code"] == "inspection_timeout"
    assert cancelled is True


@pytest.mark.asyncio
async def test_real_aiohttp_routes_authenticate_and_enforce_csrf(
    tmp_path: Path,
) -> None:
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    routes = web.RouteTableDef()
    application = web.Application()
    server_adapter = types.SimpleNamespace(routes=routes, app=application)
    models_root = tmp_path / "models"
    models_root.mkdir()
    state_directory = tmp_path / "state"
    settings = RuntimeSettings.from_env(
        {
            "SMD_ENABLED": "true",
            "SMD_PUBLIC_ORIGIN": TEST_ORIGIN,
            "SMD_AUTH_MODE": "trusted-proxy",
            "SMD_MODELS_ROOT": str(models_root),
            "SMD_STATE_DIR": str(state_directory),
        }
    )
    manager = _Manager()
    register_routes(
        server_adapter,
        settings=settings,
        signer=TokenSigner(state_directory),
        job_manager=manager,
    )
    application.add_routes(routes)
    client = TestClient(TestServer(application))
    await client.start_server()
    try:
        auth_headers = {"X-Forwarded-User": TEST_IDENTITY}
        session_response = await client.get(
            f"{API_PREFIX}/session", headers=auth_headers
        )
        assert session_response.status == 200
        session = await session_response.json()
        assert session["identity"]["email"] == TEST_IDENTITY

        denied = await client.post(
            f"{API_PREFIX}/jobs",
            headers={**auth_headers, "Origin": TEST_ORIGIN},
            json={"download_tokens": ["token"], "license_confirmed": True},
        )
        assert denied.status == 403

        accepted = await client.post(
            f"{API_PREFIX}/jobs",
            headers={
                **auth_headers,
                "Origin": TEST_ORIGIN,
                "Sec-Fetch-Site": "same-origin",
                "X-SMD-CSRF": session["csrf_token"],
            },
            json={"download_tokens": ["token"], "license_confirmed": True},
        )
        assert accepted.status == 202
        assert manager.created == [(["token"], True, TEST_SUBJECT)]
    finally:
        await client.close()
