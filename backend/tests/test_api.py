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
    MAX_BODY_BYTES,
    MAX_JOBS_PER_REQUEST,
    MAX_MODELS_PER_INSPECTION,
    ServerModelDownloaderAPI,
    register_routes,
)
from backend.metadata import MetadataError
from backend.security import ALLOWED_DIRECTORIES, SAFE_EXTENSIONS, TokenSigner


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
    ) -> None:
        if isinstance(body, bytes):
            raw = body
        elif body is None:
            raw = b""
        else:
            raw = json.dumps(body).encode()
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.content_length = len(raw) if content_length == "auto" else content_length
        self.content = _BodyStream(chunks if chunks is not None else [raw])
        self.match_info = match_info or {}


def _authority_headers(signer: TokenSigner) -> dict[str, str]:
    return {
        "X-SMD-CSRF": signer.csrf_token,
        "Origin": "https://comfy.example.com",
        "Host": "127.0.0.1:8188",
        "X-Forwarded-Host": "comfy.example.com",
        "Sec-Fetch-Site": "same-origin",
    }


class _Manager:
    def __init__(self) -> None:
        self.started = 0
        self.created: list[tuple[list[str], bool]] = []
        self.cancelled: list[str] = []
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

    async def create_jobs(self, tokens: list[str], confirmed: bool):  # noqa: ANN201
        self.created.append((tokens, confirmed))
        return [self.jobs["job-1"]]

    async def list_jobs(self):  # noqa: ANN201
        return list(self.jobs.values())

    async def get_job(self, job_id: str):  # noqa: ANN201
        return self.jobs.get(job_id)

    async def cancel_job(self, job_id: str):  # noqa: ANN201
        if job_id not in self.jobs:
            return None
        self.cancelled.append(job_id)
        self.jobs[job_id]["status"] = "cancelled"
        return self.jobs[job_id]


def _make_api(tmp_path: Path, manager: _Manager | None = None):  # noqa: ANN201
    signer = TokenSigner(tmp_path / "state")
    instance = ServerModelDownloaderAPI(
        models_root=tmp_path / "models",
        state_directory=tmp_path / "state",
        signer=signer,
        job_manager=manager,
    )
    return instance, signer


def test_session_is_no_store_and_discloses_only_capabilities(tmp_path: Path) -> None:
    api, signer = _make_api(tmp_path)
    response = _run(api.session(_Request()))
    assert response.status == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.content_type == "application/json"
    assert _response_data(response) == {
        "csrf_token": signer.csrf_token,
        "allowed_directories": list(ALLOWED_DIRECTORIES),
        "safe_extensions": list(SAFE_EXTENSIONS),
    }


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {
            "X-SMD-CSRF": "wrong",
            "Origin": "https://comfy.example.com",
            "Host": "comfy.example.com",
        },
        {
            "X-SMD-CSRF": "placeholder",
            "Origin": "https://evil.example",
            "Host": "comfy.example.com",
        },
    ],
)
def test_mutations_require_csrf_and_same_origin(
    tmp_path: Path, headers: dict[str, str]
) -> None:
    api, signer = _make_api(tmp_path, _Manager())
    if headers.get("X-SMD-CSRF") == "placeholder":
        headers["X-SMD-CSRF"] = signer.csrf_token
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
        (_Request({}, headers={"Content-Type": "text/plain"}), 415, "unsupported_media_type"),
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
        ({"download_tokens": ["signed"], "license_confirmed": False}, "license_required"),
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


def test_create_list_get_and_cancel_jobs_redact_internal_secrets(tmp_path: Path) -> None:
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
    assert manager.created == [(["signed"], True)]
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
    # One runtime manager start, regardless of how many endpoints use it.
    assert manager.started == 1


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
    def __init__(self, candidates: list[dict[str, Any]] | None = None, error: Exception | None = None):
        self.candidates = candidates or []
        self.error = error
        self.calls: list[object] = []

    async def inspect(self, session, model):  # noqa: ANN001, ANN201
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


def _candidate(
    token: str = "signed", *, sha256: str | None = None
) -> dict[str, Any]:
    return {
        "provider": "huggingface",
        "requested_name": "ae.safetensors",
        "filename": "ae.safetensors",
        "source_filename": "ae.safetensors",
        "directory": "vae",
        "relative_path": "vae/ae.safetensors",
        "canonical_url": "https://huggingface.co/org/repo/resolve/" + "a" * 40 + "/ae.safetensors",
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
    response = _run(api.inspect_models(_Request(body, headers=_authority_headers(signer))))
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


def test_register_routes_uses_existing_comfyui_http_server_only(tmp_path: Path) -> None:
    routes = _Routes()
    server = types.SimpleNamespace(routes=routes)
    registered = register_routes(
        server,
        models_root=tmp_path / "models",
        state_directory=tmp_path / "state",
        signer=TokenSigner(tmp_path / "state"),
        job_manager=_Manager(),
    )
    assert isinstance(registered, ServerModelDownloaderAPI)
    assert [(method, path) for method, path, _ in routes.registered] == [
        ("GET", f"{API_PREFIX}/session"),
        ("POST", f"{API_PREFIX}/inspect"),
        ("POST", f"{API_PREFIX}/jobs"),
        ("GET", f"{API_PREFIX}/jobs"),
        ("GET", f"{API_PREFIX}/jobs/{{job_id}}"),
        ("POST", f"{API_PREFIX}/jobs/{{job_id}}/cancel"),
    ]
