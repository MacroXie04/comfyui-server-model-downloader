from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import enum
import hashlib
import inspect
import json
import logging
import os
import re
import stat
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

try:  # ComfyUI provides aiohttp; host-side unit tests do not necessarily do so.
    import aiohttp as _aiohttp
    from aiohttp import web as _web
except ImportError:  # pragma: no cover - exercised by the dependency-free host tests.
    _aiohttp = None  # type: ignore[assignment]
    _web = None  # type: ignore[assignment]

try:  # ``server`` only exists inside a running ComfyUI process.
    from server import PromptServer as _PromptServer
except ImportError:  # pragma: no cover - normal outside ComfyUI.
    _PromptServer = None

from .metadata import MetadataError, MetadataInspector
from .safetensors_check import SafeTensorsError, validate_safetensors_file
from .security import (
    ALLOWED_DIRECTORIES,
    SAFE_EXTENSIONS,
    SecurityError,
    TokenSigner,
    require_same_origin_and_csrf,
    resolve_model_paths,
)


LOGGER = logging.getLogger(__name__)

API_PREFIX = "/server-model-downloader"
MAX_BODY_BYTES = 1024 * 1024
MAX_MODELS_PER_INSPECTION = 256
MAX_JOBS_PER_REQUEST = 20
MAX_JOB_ID_BYTES = 128
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

DEFAULT_MODELS_ROOT = Path(
    os.environ.get("SMD_MODELS_ROOT", "/srv/comfyui-data/models")
)
DEFAULT_STATE_DIRECTORY = Path(
    os.environ.get("SMD_STATE_DIR", "/srv/comfyui-data/user/server-model-downloader")
)


class APIError(ValueError):
    """An error that is safe to return to the browser."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class _FallbackJSONResponse:
    """Small response object used only by dependency-free host tests.

    Real ComfyUI requests always receive an ``aiohttp.web.Response``.
    """

    def __init__(
        self,
        data: Any,
        *,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status = status
        self.data = data
        self.text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        self.body = self.text.encode("utf-8")
        self.headers = dict(headers or {})
        self.headers.setdefault("Content-Type", "application/json; charset=utf-8")
        self.content_type = "application/json"


def _json_response(
    data: Any,
    *,
    status: int = 200,
    headers: Mapping[str, str] | None = None,
) -> Any:
    response_headers = dict(headers or {})
    if _web is None:
        return _FallbackJSONResponse(data, status=status, headers=response_headers)
    return _web.json_response(data, status=status, headers=response_headers)


def _error_response(error: APIError) -> Any:
    return _json_response(
        {"error": error.message, "code": error.code},
        status=error.status,
        headers={"Cache-Control": "no-store"},
    )


Handler = TypeVar("Handler", bound=Callable[..., Awaitable[Any]])


def _endpoint(handler: Handler) -> Handler:
    """Convert every route failure into the same JSON error envelope."""

    async def guarded(self: "ServerModelDownloaderAPI", request: Any) -> Any:
        try:
            return await handler(self, request)
        except APIError as exc:
            return _error_response(exc)
        except KeyError:
            return _error_response(APIError(404, "not_found", "download job not found"))
        except (SecurityError, MetadataError, ValueError, TypeError) as exc:
            return _error_response(APIError(400, "invalid_request", str(exc)))
        except Exception:
            LOGGER.exception("Unhandled server model downloader API error")
            return _error_response(
                APIError(500, "internal_error", "internal server error")
            )

    guarded.__name__ = handler.__name__
    guarded.__doc__ = handler.__doc__
    return guarded  # type: ignore[return-value]


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _read_json_body(request: Any) -> Any:
    """Read JSON without trusting Content-Length to enforce the 1 MiB cap."""

    content_type = str(request.headers.get("Content-Type", "")).split(";", 1)[0]
    content_type = content_type.strip().lower()
    if content_type != "application/json" and not content_type.endswith("+json"):
        raise APIError(
            415, "unsupported_media_type", "Content-Type must be application/json"
        )

    content_length = getattr(request, "content_length", None)
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except (TypeError, ValueError) as exc:
            raise APIError(400, "invalid_request", "invalid Content-Length") from exc
        if declared_length < 0:
            raise APIError(400, "invalid_request", "invalid Content-Length")
        if declared_length > MAX_BODY_BYTES:
            raise APIError(413, "body_too_large", "request body exceeds 1 MiB")

    raw = bytearray()
    stream = getattr(request, "content", None)
    if stream is not None and hasattr(stream, "iter_chunked"):
        async for chunk in stream.iter_chunked(64 * 1024):
            raw.extend(chunk)
            if len(raw) > MAX_BODY_BYTES:
                raise APIError(413, "body_too_large", "request body exceeds 1 MiB")
    elif hasattr(request, "read"):
        body = await request.read()
        raw.extend(body)
        if len(raw) > MAX_BODY_BYTES:
            raise APIError(413, "body_too_large", "request body exceeds 1 MiB")
    else:
        raise APIError(400, "invalid_request", "request body is unavailable")

    if not raw:
        raise APIError(400, "invalid_json", "request body must contain JSON")
    try:
        return json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise APIError(
            400, "invalid_json", "request body contains invalid JSON"
        ) from exc


def _validate_job_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value.encode("utf-8")) > MAX_JOB_ID_BYTES
        or not _JOB_ID_RE.fullmatch(value)
    ):
        raise APIError(400, "invalid_job_id", "invalid download job id")
    return value


def _sha256_regular_file(path: Path) -> tuple[str, int]:
    """Hash a regular file without following a final-component symlink."""

    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SecurityError("existing model is not a regular file")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            while True:
                block = stream.read(4 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                size += len(block)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest(), size


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, enum.Enum):
        return _json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        rendered = value.isoformat()
        if isinstance(value, dt.datetime) and value.tzinfo == dt.timezone.utc:
            rendered = rendered.replace("+00:00", "Z")
        return rendered
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"job response contains unsupported {type(value).__name__} value")


_PUBLIC_JOB_FIELDS = (
    "id",
    "name",
    "filename",
    "directory",
    "status",
    "bytes_downloaded",
    "size",
    "progress",
    "error",
    "sha256",
    "created_at",
    "updated_at",
    "completed_at",
)


def _job_to_dict(job: Any) -> dict[str, Any]:
    value: Any = job
    for method_name in ("public_dict", "to_public_dict", "to_dict", "as_dict"):
        method = getattr(job, method_name, None)
        if callable(method):
            value = method()
            break
    else:
        if dataclasses.is_dataclass(job) and not isinstance(job, type):
            value = {
                field: getattr(job, field)
                for field in _PUBLIC_JOB_FIELDS
                if hasattr(job, field)
            }
        elif not isinstance(job, Mapping):
            value = {
                field: getattr(job, field)
                for field in _PUBLIC_JOB_FIELDS
                if hasattr(job, field)
            }

    if not isinstance(value, Mapping):
        raise TypeError("job manager returned an invalid job record")
    # Only expose the documented job surface, even if an internal record also
    # contains a source URL, bearer token, or a signed download token.
    public = {field: value[field] for field in _PUBLIC_JOB_FIELDS if field in value}
    if "id" not in public:
        raise TypeError("job manager returned a job without an id")
    if not public.get("name") and isinstance(public.get("filename"), str):
        public["name"] = public["filename"]
    for field in ("created_at", "updated_at", "completed_at"):
        timestamp = public.get(field)
        if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
            public[field] = (
                dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
    return _json_safe(public)


def _jobs_to_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping) and "jobs" in value:
        value = value["jobs"]
    if value is None:
        return []
    if isinstance(value, Mapping) or not isinstance(value, Sequence):
        value = [value]
    return [_job_to_dict(job) for job in value]


class _Runtime:
    def __init__(
        self,
        models_root: Path,
        state_directory: Path,
        *,
        signer: TokenSigner | None = None,
        job_manager: Any | None = None,
        job_manager_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.models_root = Path(models_root).absolute()
        self.state_directory = Path(state_directory).absolute()
        self._signer = signer
        self._inspector = MetadataInspector(signer) if signer is not None else None
        self._job_manager = job_manager
        self._job_manager_factory = job_manager_factory
        self._manager_started = False
        self._manager_lock: asyncio.Lock | None = None

    @property
    def signer(self) -> TokenSigner:
        if self._signer is None:
            self._signer = TokenSigner(self.state_directory)
            self._inspector = MetadataInspector(self._signer)
        return self._signer

    @property
    def inspector(self) -> MetadataInspector:
        if self._inspector is None:
            self._inspector = MetadataInspector(self.signer)
        return self._inspector

    def _new_job_manager(self) -> Any:
        factory = self._job_manager_factory
        if factory is None:
            try:
                from .jobs import JobManager
            except ImportError as exc:
                raise APIError(
                    503, "service_unavailable", "download worker is unavailable"
                ) from exc
            factory = JobManager

        # Keep the API compatible with the concrete manager and with lightweight
        # host-test fakes. The production constructor uses these named inputs.
        parameters = inspect.signature(factory).parameters
        has_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        available = {
            "models_root": self.models_root,
            "state_directory": self.state_directory,
            "state_dir": self.state_directory,
            "signer": self.signer,
            "token_signer": self.signer,
        }
        kwargs = {
            name: value
            for name, value in available.items()
            if has_var_kwargs or name in parameters
        }
        return factory(**kwargs)

    async def job_manager(self) -> Any:
        if self._manager_lock is None:
            self._manager_lock = asyncio.Lock()
        async with self._manager_lock:
            if self._job_manager is None:
                self._job_manager = self._new_job_manager()
            if not self._manager_started:
                ensure_started = getattr(self._job_manager, "ensure_started", None)
                if not callable(ensure_started):
                    raise APIError(
                        503,
                        "service_unavailable",
                        "download worker cannot be started",
                    )
                await _maybe_await(ensure_started())
                self._manager_started = True
            return self._job_manager


class ServerModelDownloaderAPI:
    def __init__(
        self,
        *,
        models_root: Path = DEFAULT_MODELS_ROOT,
        state_directory: Path = DEFAULT_STATE_DIRECTORY,
        signer: TokenSigner | None = None,
        job_manager: Any | None = None,
        job_manager_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.runtime = _Runtime(
            models_root,
            state_directory,
            signer=signer,
            job_manager=job_manager,
            job_manager_factory=job_manager_factory,
        )

    async def _require_mutation_authority(self, request: Any) -> None:
        try:
            require_same_origin_and_csrf(
                request.headers, self.runtime.signer.csrf_token
            )
        except SecurityError as exc:
            raise APIError(403, "forbidden", str(exc)) from exc

    @_endpoint
    async def session(self, request: Any) -> Any:
        del request
        return _json_response(
            {
                "csrf_token": self.runtime.signer.csrf_token,
                "allowed_directories": list(ALLOWED_DIRECTORIES),
                "safe_extensions": list(SAFE_EXTENSIONS),
            },
            headers={"Cache-Control": "no-store"},
        )

    def _failed_model(self, model: Any, index: int, reason: str) -> dict[str, Any]:
        source = model if isinstance(model, Mapping) else {}
        name = source.get("name") if isinstance(source.get("name"), str) else ""
        url = source.get("url") if isinstance(source.get("url"), str) else ""
        directory = (
            source.get("directory") if isinstance(source.get("directory"), str) else ""
        )
        opaque_id = hashlib.sha256(
            f"{index}\0{name}\0{directory}\0{url}".encode("utf-8", "replace")
        ).hexdigest()[:24]
        return {
            "id": opaque_id,
            "name": name or f"Model {index + 1}",
            "filename": name,
            "url": url,
            "directory": directory,
            "installed": False,
            "eligible": False,
            "reason": reason,
            "source": "",
            "download_token": None,
        }

    async def _publish_candidate(
        self, candidate: Mapping[str, Any], original: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = dict(candidate)
        filename = result.get("filename")
        directory = result.get("directory")
        expected_size = result.get("size")
        if not isinstance(filename, str) or not isinstance(directory, str):
            raise SecurityError("inspected model is missing a safe destination")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size <= 0
        ):
            raise SecurityError("inspected model is missing a valid size")
        try:
            final_path, _ = resolve_model_paths(
                self.runtime.models_root, directory, filename
            )
            try:
                file_stat = os.lstat(final_path)
            except FileNotFoundError:
                installed = False
                conflicting_file = False
            else:
                candidate_file = (
                    stat.S_ISREG(file_stat.st_mode)
                    and not stat.S_ISLNK(file_stat.st_mode)
                    and file_stat.st_size == expected_size
                )
                installed = False
                conflicting_file = not candidate_file
                verification_reason = (
                    "existing file has an unsafe type or different size"
                    if not candidate_file
                    else ""
                )
                if candidate_file:
                    expected_sha256 = result.get("sha256")
                    if not isinstance(expected_sha256, str):
                        conflicting_file = True
                        verification_reason = "provider metadata is missing SHA256"
                    else:
                        try:
                            actual_sha256, actual_size = await asyncio.to_thread(
                                _sha256_regular_file, final_path
                            )
                            if (
                                actual_size != expected_size
                                or actual_sha256.lower() != expected_sha256.lower()
                            ):
                                raise SecurityError(
                                    "existing file failed SHA256 verification"
                                )
                            await asyncio.to_thread(
                                validate_safetensors_file, final_path
                            )
                        except (OSError, SecurityError, SafeTensorsError) as exc:
                            conflicting_file = True
                            verification_reason = str(exc)
                        else:
                            installed = True
        except SecurityError as exc:
            result.update(
                {
                    "installed": False,
                    "eligible": False,
                    "reason": str(exc),
                    "download_token": None,
                }
            )
        else:
            if conflicting_file:
                result.update(
                    {
                        "installed": False,
                        "eligible": False,
                        "reason": (
                            "an existing file could not be verified"
                            + (f": {verification_reason}" if verification_reason else "")
                            + "; move or remove it manually before downloading"
                        ),
                        "download_token": None,
                    }
                )
            else:
                result.update(
                    {
                        "installed": installed,
                        "eligible": not installed,
                        "reason": "model is already installed" if installed else "",
                    }
                )
            if installed or conflicting_file:
                result["download_token"] = None

        result["name"] = result.get("requested_name") or filename
        # Retain the graph URL so the frontend can associate the response with
        # the node that supplied it; canonical_url remains server-authoritative.
        result["url"] = original.get("url", "")
        provider = result.get("provider")
        result["source"] = (
            "Hugging Face"
            if provider == "huggingface"
            else "Civitai"
            if provider == "civitai"
            else ""
        )
        model_id_material = (
            f"{result.get('relative_path', '')}\0{result.get('revision', '')}"
            f"\0{result.get('sha256', '')}"
        )
        result["id"] = hashlib.sha256(
            model_id_material.encode("utf-8", "replace")
        ).hexdigest()[:24]
        return result

    @_endpoint
    async def inspect_models(self, request: Any) -> Any:
        await self._require_mutation_authority(request)
        body = await _read_json_body(request)
        if not isinstance(body, Mapping):
            raise APIError(400, "invalid_request", "JSON body must be an object")
        models = body.get("models")
        if not isinstance(models, list):
            raise APIError(400, "invalid_request", "models must be a list")
        if len(models) > MAX_MODELS_PER_INSPECTION:
            raise APIError(
                400,
                "too_many_models",
                f"at most {MAX_MODELS_PER_INSPECTION} models may be inspected",
            )
        if not models:
            return _json_response(
                {"models": []}, headers={"Cache-Control": "no-store"}
            )
        if _aiohttp is None:
            raise APIError(
                503, "service_unavailable", "aiohttp is unavailable on this server"
            )

        timeout = _aiohttp.ClientTimeout(
            total=90, connect=10, sock_connect=10, sock_read=30
        )
        inspected: list[dict[str, Any]] = []
        async with _aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
            for index, model in enumerate(models):
                try:
                    candidates = await self.runtime.inspector.inspect(session, model)
                    if not isinstance(model, Mapping):
                        raise MetadataError("each model must be an object")
                    for candidate in candidates:
                        inspected.append(
                            await self._publish_candidate(candidate, model)
                        )
                except (SecurityError, MetadataError, TypeError, ValueError) as exc:
                    inspected.append(self._failed_model(model, index, str(exc)))
                except Exception:
                    LOGGER.exception("Model metadata inspection failed")
                    inspected.append(
                        self._failed_model(
                            model, index, "provider metadata inspection failed"
                        )
                    )
        return _json_response(
            {"models": inspected}, headers={"Cache-Control": "no-store"}
        )

    @_endpoint
    async def create_jobs(self, request: Any) -> Any:
        await self._require_mutation_authority(request)
        body = await _read_json_body(request)
        if not isinstance(body, Mapping):
            raise APIError(400, "invalid_request", "JSON body must be an object")
        tokens = body.get("download_tokens")
        if not isinstance(tokens, list) or not tokens:
            raise APIError(
                400, "invalid_request", "download_tokens must be a non-empty list"
            )
        if len(tokens) > MAX_JOBS_PER_REQUEST:
            raise APIError(
                400,
                "too_many_jobs",
                f"at most {MAX_JOBS_PER_REQUEST} downloads may be submitted",
            )
        if any(not isinstance(token, str) or not token for token in tokens):
            raise APIError(
                400, "invalid_request", "every download token must be a string"
            )
        if body.get("license_confirmed") is not True:
            raise APIError(
                400,
                "license_required",
                "license_confirmed must be true for every submitted batch",
            )

        manager = await self.runtime.job_manager()
        jobs = await _maybe_await(manager.create_jobs(tokens, True))
        return _json_response(
            {"jobs": _jobs_to_list(jobs)},
            status=202,
            headers={"Cache-Control": "no-store"},
        )

    @_endpoint
    async def list_jobs(self, request: Any) -> Any:
        del request
        manager = await self.runtime.job_manager()
        jobs = await _maybe_await(manager.list_jobs())
        return _json_response(
            {"jobs": _jobs_to_list(jobs)}, headers={"Cache-Control": "no-store"}
        )

    @_endpoint
    async def get_job(self, request: Any) -> Any:
        job_id = _validate_job_id(request.match_info.get("job_id"))
        manager = await self.runtime.job_manager()
        try:
            job = await _maybe_await(manager.get_job(job_id))
        except ValueError as exc:
            if "not found" in str(exc).lower():
                raise APIError(404, "not_found", "download job not found") from exc
            raise
        if job is None:
            raise APIError(404, "not_found", "download job not found")
        return _json_response(
            {"job": _job_to_dict(job)}, headers={"Cache-Control": "no-store"}
        )

    @_endpoint
    async def cancel_job(self, request: Any) -> Any:
        await self._require_mutation_authority(request)
        body = await _read_json_body(request)
        if not isinstance(body, Mapping):
            raise APIError(400, "invalid_request", "JSON body must be an object")
        job_id = _validate_job_id(request.match_info.get("job_id"))
        manager = await self.runtime.job_manager()
        try:
            job = await _maybe_await(manager.cancel_job(job_id))
        except ValueError as exc:
            if "not found" in str(exc).lower():
                raise APIError(404, "not_found", "download job not found") from exc
            raise
        if job is None or job is False:
            raise APIError(404, "not_found", "download job not found")
        if job is True:
            job = await _maybe_await(manager.get_job(job_id))
        if job is None:
            raise APIError(404, "not_found", "download job not found")
        return _json_response(
            {"job": _job_to_dict(job)},
            status=202,
            headers={"Cache-Control": "no-store"},
        )


def register_routes(
    prompt_server: Any | None = None,
    *,
    models_root: Path = DEFAULT_MODELS_ROOT,
    state_directory: Path = DEFAULT_STATE_DIRECTORY,
    signer: TokenSigner | None = None,
    job_manager: Any | None = None,
    job_manager_factory: Callable[..., Any] | None = None,
) -> ServerModelDownloaderAPI:
    """Register the downloader routes on ComfyUI's existing HTTP server."""

    if prompt_server is None:
        if _PromptServer is None or getattr(_PromptServer, "instance", None) is None:
            raise RuntimeError("ComfyUI PromptServer is unavailable")
        prompt_server = _PromptServer.instance
    routes = getattr(prompt_server, "routes", prompt_server)
    if not callable(getattr(routes, "get", None)) or not callable(
        getattr(routes, "post", None)
    ):
        raise TypeError("PromptServer does not expose an aiohttp route table")

    api = ServerModelDownloaderAPI(
        models_root=models_root,
        state_directory=state_directory,
        signer=signer,
        job_manager=job_manager,
        job_manager_factory=job_manager_factory,
    )
    routes.get(f"{API_PREFIX}/session")(api.session)
    routes.post(f"{API_PREFIX}/inspect")(api.inspect_models)
    routes.post(f"{API_PREFIX}/jobs")(api.create_jobs)
    routes.get(f"{API_PREFIX}/jobs")(api.list_jobs)
    routes.get(f"{API_PREFIX}/jobs/{{job_id}}")(api.get_job)
    routes.post(f"{API_PREFIX}/jobs/{{job_id}}/cancel")(api.cancel_job)
    return api


REGISTERED_API: ServerModelDownloaderAPI | None = None
if (
    _aiohttp is not None
    and _PromptServer is not None
    and getattr(_PromptServer, "instance", None) is not None
):  # pragma: no cover - only runs inside ComfyUI.
    REGISTERED_API = register_routes(_PromptServer.instance)


__all__ = [
    "API_PREFIX",
    "DEFAULT_MODELS_ROOT",
    "DEFAULT_STATE_DIRECTORY",
    "MAX_BODY_BYTES",
    "REGISTERED_API",
    "ServerModelDownloaderAPI",
    "register_routes",
]
