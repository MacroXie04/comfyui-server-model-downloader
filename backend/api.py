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
import time
import uuid
from collections import deque
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

from .auth import (
    AuthenticationError,
    Authenticator,
    Identity,
)
from .jobs import JobError
from .metadata import MetadataError, MetadataInspector
from .safetensors_check import SafeTensorsError, validate_safetensors_file
from .security import (
    ALLOWED_DIRECTORIES,
    SAFE_EXTENSIONS,
    SecurityError,
    TokenSigner,
    require_subject_origin_and_csrf,
    resolve_model_paths,
)
from .settings import RuntimeSettings, SettingsState, load_runtime_settings

LOGGER = logging.getLogger(__name__)

API_PREFIX = "/server-model-downloader"
API_COMPAT_PREFIX = f"/api{API_PREFIX}"
API_VERSION = "1"
EXTENSION_VERSION = "1.0.0"
MAX_BODY_BYTES = 1024 * 1024
MAX_MODELS_PER_INSPECTION = 50
MAX_JOBS_PER_REQUEST = 20
MAX_JOB_ID_BYTES = 128
INSPECTION_CONCURRENCY = 4
INSPECTION_DEADLINE_SECONDS = 60
INSPECTIONS_PER_IDENTITY_PER_MINUTE = 10
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


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


def _error_response(error: APIError, request_id: str) -> Any:
    return _json_response(
        {"error": error.message, "code": error.code, "request_id": request_id},
        status=error.status,
        headers={"Cache-Control": "no-store", "X-Request-ID": request_id},
    )


Handler = TypeVar("Handler", bound=Callable[..., Awaitable[Any]])


def _endpoint(handler: Handler) -> Handler:
    """Convert every route failure into the same JSON error envelope."""

    async def guarded(self: ServerModelDownloaderAPI, request: Any) -> Any:
        request_id = uuid.uuid4().hex
        try:
            response = await handler(self, request)
            response.headers["X-Request-ID"] = request_id
            return response
        except APIError as exc:
            return _error_response(exc, request_id)
        except AuthenticationError as exc:
            return _error_response(APIError(exc.status, exc.code, str(exc)), request_id)
        except KeyError:
            return _error_response(
                APIError(404, "not_found", "download job not found"), request_id
            )
        except (SecurityError, MetadataError, JobError) as exc:
            return _error_response(
                APIError(400, "invalid_request", str(exc)), request_id
            )
        except (ValueError, TypeError):
            return _error_response(
                APIError(400, "invalid_request", "invalid request"), request_id
            )
        except Exception as exc:
            # Do not emit exception messages or tracebacks: OSError strings can
            # contain absolute paths and upstream libraries may embed secrets.
            LOGGER.error(
                "Unhandled Server Model Downloader API error type=%s request_id=%s",
                type(exc).__name__,
                request_id,
            )
            return _error_response(
                APIError(500, "internal_error", "internal server error"),
                request_id,
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
    "phase",
    "error_code",
    "attempt",
    "max_attempts",
    "cancel_requested",
)


def _job_to_dict(job: Any) -> dict[str, Any]:
    value: Any = job
    for method_name in ("public_dict", "to_public_dict", "to_dict", "as_dict"):
        method = getattr(job, method_name, None)
        if callable(method):
            value = method()
            break
    else:
        if (
            dataclasses.is_dataclass(job)
            and not isinstance(job, type)
            or not isinstance(job, Mapping)
        ):
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


class _InspectionRateLimiter:
    """Small in-memory per-identity limiter for provider metadata scans."""

    def __init__(
        self,
        limit: int = INSPECTIONS_PER_IDENTITY_PER_MINUTE,
        window_seconds: float = 60.0,
    ) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = {}
        self._lock: asyncio.Lock | None = None

    async def require_capacity(self, subject: str) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        now = time.monotonic()
        async with self._lock:
            events = self._events.setdefault(subject, deque())
            cutoff = now - self.window_seconds
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                raise APIError(
                    429,
                    "rate_limited",
                    "too many model inspections; retry later",
                )
            events.append(now)
            if len(self._events) > 1024:
                stale: list[str] = []
                for key, values in self._events.items():
                    while values and values[0] <= cutoff:
                        values.popleft()
                    if not values:
                        stale.append(key)
                for key in stale:
                    self._events.pop(key, None)
                while len(self._events) > 1024:
                    oldest = min(
                        self._events,
                        key=lambda key: self._events[key][-1],
                    )
                    self._events.pop(oldest, None)


class _Runtime:
    def __init__(
        self,
        settings_state: SettingsState,
        *,
        signer: TokenSigner | None = None,
        authenticator: Authenticator | None = None,
        job_manager: Any | None = None,
        job_manager_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings_state = settings_state
        self.settings = settings_state.settings
        self._signer = signer
        self._authenticator = authenticator
        self._inspector = MetadataInspector(signer) if signer is not None else None
        self._job_manager = job_manager
        self._job_manager_factory = job_manager_factory
        self._manager_lock: asyncio.Lock | None = None
        self._startup_error: str | None = None

    @property
    def models_root(self) -> Path:
        if self.settings is None or self.settings.models_root is None:
            raise APIError(503, "service_unavailable", "downloader is unavailable")
        return self.settings.models_root

    @property
    def state_directory(self) -> Path:
        if self.settings is None or self.settings.state_directory is None:
            raise APIError(503, "service_unavailable", "downloader is unavailable")
        return self.settings.state_directory

    def require_configured(self) -> RuntimeSettings:
        if self.settings_state.error is not None:
            raise APIError(
                503,
                "configuration_error",
                "downloader configuration is invalid",
            )
        if self.settings is None or not self.settings.enabled:
            raise APIError(503, "service_disabled", "downloader is disabled")
        return self.settings

    def require_operational(self) -> RuntimeSettings:
        settings = self.require_configured()
        if self._startup_error is not None:
            raise APIError(
                503,
                "service_degraded",
                "download worker is degraded",
            )
        manager = self._job_manager
        health = getattr(manager, "health", None) if manager is not None else None
        if callable(health):
            try:
                state = health()
            except Exception:
                self._startup_error = "manager health check failed"
                raise APIError(
                    503,
                    "service_degraded",
                    "download worker is degraded",
                ) from None
            if isinstance(state, Mapping) and state.get("status") == "degraded":
                self._startup_error = "manager reported degraded"
                raise APIError(
                    503,
                    "service_degraded",
                    "download worker is degraded",
                )
        return settings

    @property
    def signer(self) -> TokenSigner:
        self.require_configured()
        if self._signer is None:
            self._signer = TokenSigner(self.state_directory)
            self._inspector = MetadataInspector(self._signer)
        return self._signer

    @property
    def authenticator(self) -> Authenticator:
        settings = self.require_configured()
        if self._authenticator is None:
            self._authenticator = Authenticator(settings)
        return self._authenticator

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
        self.require_operational()
        if self._manager_lock is None:
            self._manager_lock = asyncio.Lock()
        async with self._manager_lock:
            if self._job_manager is None:
                try:
                    self._job_manager = self._new_job_manager()
                except (JobError, OSError, SecurityError) as exc:
                    self._startup_error = type(exc).__name__
                    raise APIError(
                        503,
                        "service_degraded",
                        "download worker could not be initialized",
                    ) from exc
            ensure_started = getattr(self._job_manager, "ensure_started", None)
            if not callable(ensure_started):
                self._startup_error = "missing worker lifecycle"
                raise APIError(
                    503,
                    "service_degraded",
                    "download worker cannot be started",
                )
            try:
                await _maybe_await(ensure_started())
            except (JobError, OSError, SecurityError) as exc:
                self._startup_error = type(exc).__name__
                raise APIError(
                    503,
                    "service_degraded",
                    "download worker could not be started",
                ) from exc
            return self._job_manager

    async def startup(self) -> None:
        try:
            self.require_configured()
            await self.job_manager()
        except (APIError, AuthenticationError, OSError, SecurityError, JobError) as exc:
            if isinstance(exc, APIError) and exc.code in {
                "service_disabled",
                "configuration_error",
            }:
                return
            self._startup_error = type(exc).__name__
            LOGGER.error(
                "Server Model Downloader entered degraded state during startup (%s)",
                type(exc).__name__,
            )

    async def shutdown(self) -> None:
        manager = self._job_manager
        if manager is None:
            return
        stop = getattr(manager, "stop", None)
        if not callable(stop):
            return
        try:
            await asyncio.wait_for(_maybe_await(stop()), timeout=30)
        except asyncio.TimeoutError:
            LOGGER.error("Server Model Downloader shutdown timed out")
        except Exception as exc:
            LOGGER.error(
                "Server Model Downloader shutdown failed (%s)", type(exc).__name__
            )

    def health(self) -> tuple[dict[str, Any], int]:
        if self.settings_state.error is not None:
            return {
                "status": "degraded",
                "state": "configuration-error",
                "reason": "downloader configuration is invalid",
            }, 503
        if self.settings is None or not self.settings.enabled:
            return {
                "status": "degraded",
                "state": "disabled",
                "reason": "downloader is disabled",
            }, 503
        if self._startup_error is not None:
            return {
                "status": "degraded",
                "state": "degraded",
                "reason": "download worker is unavailable",
            }, 503
        manager = self._job_manager
        health = getattr(manager, "health", None) if manager is not None else None
        if callable(health):
            value = health()
            if isinstance(value, Mapping):
                document = {
                    key: value[key]
                    for key in (
                        "status",
                        "state",
                        "worker_running",
                        "instance_lock_acquired",
                        "queued_jobs",
                        "quarantined_state",
                    )
                    if key in value
                }
                if value.get("reason") or value.get("last_worker_error"):
                    document["reason"] = "download worker is degraded"
                status = 503 if document.get("status") == "degraded" else 200
                return document, status
        return {"status": "ok", "state": "ready"}, 200


class ServerModelDownloaderAPI:
    def __init__(
        self,
        *,
        settings_state: SettingsState | None = None,
        settings: RuntimeSettings | None = None,
        signer: TokenSigner | None = None,
        authenticator: Authenticator | None = None,
        job_manager: Any | None = None,
        job_manager_factory: Callable[..., Any] | None = None,
    ) -> None:
        if settings_state is not None and settings is not None:
            raise TypeError("provide settings or settings_state, not both")
        if settings_state is None:
            settings_state = (
                SettingsState(settings)
                if settings is not None
                else load_runtime_settings()
            )
        self.runtime = _Runtime(
            settings_state,
            signer=signer,
            authenticator=authenticator,
            job_manager=job_manager,
            job_manager_factory=job_manager_factory,
        )
        self._inspection_semaphore = asyncio.Semaphore(INSPECTION_CONCURRENCY)
        self._inspection_rate_limiter = _InspectionRateLimiter()

    async def _authenticate(self, request: Any) -> Identity:
        self.runtime.require_configured()
        return await self.runtime.authenticator.authenticate_request(request)

    async def _require_mutation_authority(
        self, request: Any, identity: Identity
    ) -> None:
        settings = self.runtime.require_operational()
        assert settings.public_origin is not None
        try:
            require_subject_origin_and_csrf(
                request.headers,
                self.runtime.signer,
                identity.subject,
                settings.public_origin,
            )
        except SecurityError as exc:
            raise APIError(403, "forbidden", "request authorization failed") from exc

    @_endpoint
    async def session(self, request: Any) -> Any:
        identity = await self._authenticate(request)
        settings = self.runtime.require_operational()
        assert settings.public_origin is not None
        csrf_token, csrf_expires_at = self.runtime.signer.issue_csrf(
            identity.subject, settings.public_origin
        )
        return _json_response(
            {
                "api_version": API_VERSION,
                "extension_version": EXTENSION_VERSION,
                "csrf_token": csrf_token,
                "csrf_expires_at": csrf_expires_at,
                "allowed_directories": list(ALLOWED_DIRECTORIES),
                "safe_extensions": list(SAFE_EXTENSIONS),
                "capabilities": {
                    "providers": ["huggingface", "civitai"],
                    "resume": True,
                    "sha256": True,
                    "discard_partial": True,
                    "max_models_per_scan": MAX_MODELS_PER_INSPECTION,
                    "single_process": True,
                },
                "identity": identity.as_public_dict(),
            },
            headers={"Cache-Control": "no-store"},
        )

    @_endpoint
    async def health(self, request: Any) -> Any:
        identity = await self._authenticate(request)
        del identity
        document, status = self.runtime.health()
        return _json_response(
            {
                "api_version": API_VERSION,
                "extension_version": EXTENSION_VERSION,
                **document,
            },
            status=status,
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
                        except OSError:
                            conflicting_file = True
                            verification_reason = "filesystem verification failed"
                        except (SecurityError, SafeTensorsError) as exc:
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
                            + (
                                f": {verification_reason}"
                                if verification_reason
                                else ""
                            )
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

    async def _inspect_one(
        self,
        session: Any,
        model: Any,
        index: int,
        identity: Identity,
    ) -> list[dict[str, Any]]:
        async with self._inspection_semaphore:
            try:
                if not isinstance(model, Mapping):
                    raise MetadataError("each model must be an object")
                candidates = await self.runtime.inspector.inspect(
                    session,
                    model,
                    subject=identity.subject,
                )
                return [
                    await self._publish_candidate(candidate, model)
                    for candidate in candidates
                ]
            except (SecurityError, MetadataError) as exc:
                return [self._failed_model(model, index, str(exc))]
            except (TypeError, ValueError):
                return [
                    self._failed_model(
                        model,
                        index,
                        "model metadata is invalid",
                    )
                ]
            except Exception as exc:
                LOGGER.error(
                    "Model metadata inspection failed (%s)", type(exc).__name__
                )
                return [
                    self._failed_model(
                        model,
                        index,
                        "provider metadata inspection failed",
                    )
                ]

    @_endpoint
    async def inspect_models(self, request: Any) -> Any:
        identity = await self._authenticate(request)
        await self._require_mutation_authority(request, identity)
        await self._inspection_rate_limiter.require_capacity(identity.subject)
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
            return _json_response({"models": []}, headers={"Cache-Control": "no-store"})
        if _aiohttp is None:
            raise APIError(
                503, "service_unavailable", "aiohttp is unavailable on this server"
            )

        timeout = _aiohttp.ClientTimeout(
            total=INSPECTION_DEADLINE_SECONDS,
            connect=10,
            sock_connect=10,
            sock_read=30,
        )
        async with _aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
            tasks = [
                asyncio.create_task(self._inspect_one(session, model, index, identity))
                for index, model in enumerate(models)
            ]
            try:
                groups = await asyncio.wait_for(
                    asyncio.gather(*tasks),
                    timeout=INSPECTION_DEADLINE_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise APIError(
                    504,
                    "inspection_timeout",
                    "model inspection exceeded its 60 second deadline",
                ) from exc
        inspected = [item for group in groups for item in group]
        return _json_response(
            {"models": inspected}, headers={"Cache-Control": "no-store"}
        )

    @_endpoint
    async def create_jobs(self, request: Any) -> Any:
        identity = await self._authenticate(request)
        await self._require_mutation_authority(request, identity)
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
        jobs = await _maybe_await(manager.create_jobs(tokens, True, identity.subject))
        return _json_response(
            {"jobs": _jobs_to_list(jobs)},
            status=202,
            headers={"Cache-Control": "no-store"},
        )

    @_endpoint
    async def list_jobs(self, request: Any) -> Any:
        identity = await self._authenticate(request)
        manager = await self.runtime.job_manager()
        query = getattr(request, "query", {})
        raw_limit = query.get("limit", "50") if isinstance(query, Mapping) else "50"
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError) as exc:
            raise APIError(400, "invalid_limit", "limit must be an integer") from exc
        cursor = query.get("cursor") if isinstance(query, Mapping) else None
        if cursor is not None and not isinstance(cursor, str):
            raise APIError(400, "invalid_cursor", "cursor must be a string")
        history = getattr(manager, "list_job_history", None)
        if callable(history):
            page = await _maybe_await(
                history(identity.subject, cursor=cursor, limit=limit)
            )
        else:
            page = {
                "jobs": await _maybe_await(manager.list_jobs(identity.subject)),
                "next_cursor": None,
            }
        if not isinstance(page, Mapping):
            raise TypeError("job manager returned an invalid history page")
        return _json_response(
            {
                "jobs": _jobs_to_list(page.get("jobs")),
                "next_cursor": page.get("next_cursor"),
            },
            headers={"Cache-Control": "no-store"},
        )

    @_endpoint
    async def get_job(self, request: Any) -> Any:
        identity = await self._authenticate(request)
        job_id = _validate_job_id(request.match_info.get("job_id"))
        manager = await self.runtime.job_manager()
        try:
            job = await _maybe_await(manager.get_job(job_id, identity.subject))
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
        identity = await self._authenticate(request)
        await self._require_mutation_authority(request, identity)
        body = await _read_json_body(request)
        if not isinstance(body, Mapping):
            raise APIError(400, "invalid_request", "JSON body must be an object")
        job_id = _validate_job_id(request.match_info.get("job_id"))
        manager = await self.runtime.job_manager()
        try:
            job = await _maybe_await(manager.cancel_job(job_id, identity.subject))
        except ValueError as exc:
            if "not found" in str(exc).lower():
                raise APIError(404, "not_found", "download job not found") from exc
            raise
        if job is None or job is False:
            raise APIError(404, "not_found", "download job not found")
        if job is True:
            job = await _maybe_await(manager.get_job(job_id, identity.subject))
        if job is None:
            raise APIError(404, "not_found", "download job not found")
        return _json_response(
            {"job": _job_to_dict(job)},
            status=202,
            headers={"Cache-Control": "no-store"},
        )

    @_endpoint
    async def discard_partial(self, request: Any) -> Any:
        identity = await self._authenticate(request)
        await self._require_mutation_authority(request, identity)
        job_id = _validate_job_id(request.match_info.get("job_id"))
        manager = await self.runtime.job_manager()
        discard = getattr(manager, "discard_partial", None)
        if not callable(discard):
            raise APIError(
                503,
                "service_degraded",
                "partial cleanup is unavailable",
            )
        try:
            job = await _maybe_await(discard(job_id, identity.subject))
        except KeyError as exc:
            raise APIError(404, "not_found", "download job not found") from exc
        return _json_response(
            {"job": _job_to_dict(job)},
            headers={"Cache-Control": "no-store"},
        )


def register_routes(
    prompt_server: Any | None = None,
    *,
    settings_state: SettingsState | None = None,
    settings: RuntimeSettings | None = None,
    signer: TokenSigner | None = None,
    authenticator: Authenticator | None = None,
    job_manager: Any | None = None,
    job_manager_factory: Callable[..., Any] | None = None,
) -> ServerModelDownloaderAPI:
    """Register the downloader routes on ComfyUI's existing HTTP server."""

    if prompt_server is None:
        if _PromptServer is None or getattr(_PromptServer, "instance", None) is None:
            raise RuntimeError("ComfyUI PromptServer is unavailable")
        prompt_server = _PromptServer.instance
    existing = getattr(prompt_server, "_server_model_downloader_api", None)
    if isinstance(existing, ServerModelDownloaderAPI):
        return existing
    routes = getattr(prompt_server, "routes", prompt_server)
    if (
        not callable(getattr(routes, "get", None))
        or not callable(getattr(routes, "post", None))
        or not callable(getattr(routes, "delete", None))
    ):
        raise TypeError("PromptServer does not expose an aiohttp route table")

    api = ServerModelDownloaderAPI(
        settings_state=settings_state,
        settings=settings,
        signer=signer,
        authenticator=authenticator,
        job_manager=job_manager,
        job_manager_factory=job_manager_factory,
    )
    routes.get(f"{API_PREFIX}/health")(api.health)
    routes.get(f"{API_PREFIX}/session")(api.session)
    routes.post(f"{API_PREFIX}/inspect")(api.inspect_models)
    routes.post(f"{API_PREFIX}/jobs")(api.create_jobs)
    routes.get(f"{API_PREFIX}/jobs")(api.list_jobs)
    routes.get(f"{API_PREFIX}/jobs/{{job_id}}")(api.get_job)
    routes.post(f"{API_PREFIX}/jobs/{{job_id}}/cancel")(api.cancel_job)
    routes.delete(f"{API_PREFIX}/jobs/{{job_id}}/partial")(api.discard_partial)
    prompt_server._server_model_downloader_api = api

    app = getattr(prompt_server, "app", None)
    on_startup = getattr(app, "on_startup", None)
    on_cleanup = getattr(app, "on_cleanup", None)
    if on_startup is not None and hasattr(on_startup, "append"):

        async def start_downloader(application: Any) -> None:
            del application
            await api.runtime.startup()

        on_startup.append(start_downloader)
    if on_cleanup is not None and hasattr(on_cleanup, "append"):

        async def stop_downloader(application: Any) -> None:
            del application
            await api.runtime.shutdown()

        on_cleanup.append(stop_downloader)
    return api


REGISTERED_API: ServerModelDownloaderAPI | None = None
if (
    _aiohttp is not None
    and _PromptServer is not None
    and getattr(_PromptServer, "instance", None) is not None
):  # pragma: no cover - only runs inside ComfyUI.
    REGISTERED_API = register_routes(_PromptServer.instance)


__all__ = [
    "API_COMPAT_PREFIX",
    "API_PREFIX",
    "API_VERSION",
    "EXTENSION_VERSION",
    "MAX_BODY_BYTES",
    "REGISTERED_API",
    "ServerModelDownloaderAPI",
    "register_routes",
]
