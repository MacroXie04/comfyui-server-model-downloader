from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import enum
import hashlib
import hmac
import ipaddress
import json
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiohttp
import jwt
import pytest

from backend import api, auth, security, settings
from backend.auth import (
    AuthenticationError,
    AuthenticationUnavailable,
    Authenticator,
    CloudflareAccessVerifier,
    CloudflareJWKSCache,
)
from backend.job_store import (
    MAX_HISTORY_AGE_SECONDS,
    MAX_TERMINAL_HISTORY,
    JobStoreError,
    PersistentJobStore,
    StateDirectoryLease,
    decode_cursor,
    encode_cursor,
    migrate_job,
)
from backend.security import SecurityError, TokenSigner
from backend.settings import AuthMode, RuntimeSettings, SettingsError
from backend.tests.test_auth import _keypair


class _Chunks:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def iter_chunked(self, size: int):  # noqa: ANN201
        del size
        for chunk in self.chunks:
            yield chunk


class _JWKSResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self.content = _Chunks(chunks or [b'{"keys": []}'])

    async def __aenter__(self):  # noqa: ANN204
        return self

    async def __aexit__(self, *args: Any) -> None:
        del args


class _JWKSSession:
    response: _JWKSResponse | BaseException

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    async def __aenter__(self):  # noqa: ANN204
        return self

    async def __aexit__(self, *args: Any) -> None:
        del args

    def get(self, *args: Any, **kwargs: Any) -> _JWKSResponse:
        del args, kwargs
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        _JWKSResponse(status=503),
        _JWKSResponse(headers={"Content-Length": str(auth.MAX_JWKS_BYTES + 1)}),
        _JWKSResponse(headers={"Content-Length": "invalid"}),
        _JWKSResponse(chunks=[b"x" * (auth.MAX_JWKS_BYTES + 1)]),
        _JWKSResponse(chunks=[b"not-json"]),
        _JWKSResponse(chunks=[b"[]"]),
        OSError("network unavailable"),
    ],
)
async def test_default_jwks_fetcher_fails_closed_for_bad_responses(
    monkeypatch: pytest.MonkeyPatch,
    response: _JWKSResponse | BaseException,
) -> None:
    _JWKSSession.response = response
    monkeypatch.setattr(aiohttp, "ClientSession", _JWKSSession)
    with pytest.raises(AuthenticationUnavailable):
        await auth._default_jwks_fetcher("https://team.example/certs")


@pytest.mark.asyncio
async def test_default_jwks_fetcher_accepts_a_bounded_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _JWKSSession.response = _JWKSResponse(
        headers={"Content-Length": "12"}, chunks=[b'{"keys":', b" []}"]
    )
    monkeypatch.setattr(aiohttp, "ClientSession", _JWKSSession)
    assert await auth._default_jwks_fetcher("https://team.example/certs") == {
        "keys": []
    }


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"keys": []},
        {"keys": [None]},
        {"keys": [{"kid": None}]},
        {"keys": [{"kid": "skip", "kty": "EC"}]},
        {"keys": [{"kid": "skip", "kty": "RSA", "alg": "ES256"}]},
        {"keys": [{"kid": "skip", "kty": "RSA", "use": "enc"}]},
        {"keys": [{"kid": "bad", "kty": "RSA", "n": "!", "e": "!"}]},
    ],
)
def test_jwks_parser_rejects_malformed_or_unusable_sets(
    document: dict[str, Any],
) -> None:
    with pytest.raises(AuthenticationUnavailable):
        CloudflareJWKSCache._parse_keys(document)


def test_jwks_parser_rejects_duplicate_key_ids() -> None:
    _, public_jwk = _keypair("duplicate")
    with pytest.raises(AuthenticationUnavailable):
        CloudflareJWKSCache._parse_keys({"keys": [public_jwk, public_jwk]})


def test_jwks_cache_validates_constructor_and_kid() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        CloudflareJWKSCache("http://team.example/certs")
    with pytest.raises(ValueError, match="positive"):
        CloudflareJWKSCache("https://team.example/certs", ttl_seconds=0)


@pytest.mark.asyncio
async def test_jwks_cache_wraps_unexpected_fetcher_errors() -> None:
    async def explode(url: str) -> dict[str, Any]:
        del url
        raise RuntimeError("internal detail")

    cache = CloudflareJWKSCache("https://team.example/certs", fetcher=explode)
    with pytest.raises(AuthenticationUnavailable):
        await cache.get_key("kid")
    for invalid in ("", "x" * 513, None):
        with pytest.raises(AuthenticationError) as caught:
            await cache.get_key(invalid)  # type: ignore[arg-type]
        assert caught.value.code == "invalid_token"


@pytest.mark.asyncio
async def test_jwks_cache_suppresses_repeated_unknown_kid_refreshes() -> None:
    _, public_jwk = _keypair("known")
    calls: list[str] = []

    async def fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        return {"keys": [public_jwk]}

    cache = CloudflareJWKSCache(
        "https://team.example/certs",
        fetcher=fetch,
        min_refresh_interval_seconds=60,
    )
    assert await cache.get_key("known") is not None
    with pytest.raises(AuthenticationError):
        await cache.get_key("unknown")
    assert len(calls) == 1
    assert cache._lock_for_loop() is cache._lock


class _StaticKeyCache:
    async def get_key(self, kid: str) -> object:
        del kid
        return object()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claims",
    [
        [],
        {"exp": "invalid", "sub": "user"},
        {"exp": 1, "nbf": True, "sub": "user"},
        {"exp": 1, "sub": None},
        {"exp": 1, "sub": ""},
        {"exp": 1, "sub": "x" * 1025},
        {"exp": 1, "sub": "user\nname"},
        {"exp": 1, "sub": "user", "email": 3},
        {"exp": 1, "sub": "user", "email": ""},
        {"exp": 1, "sub": "user", "email": "x" * 321},
        {"exp": 1, "sub": "user", "email": "bad email@example.com"},
    ],
)
async def test_cloudflare_verifier_rejects_invalid_decoded_claim_shapes(
    monkeypatch: pytest.MonkeyPatch, claims: Any
) -> None:
    verifier = CloudflareAccessVerifier(
        "https://team.cloudflareaccess.com",
        ("audience",),
        _StaticKeyCache(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        jwt, "get_unverified_header", lambda token: {"alg": "RS256", "kid": "k"}
    )
    monkeypatch.setattr(jwt, "decode", lambda *args, **kwargs: claims)
    with pytest.raises(AuthenticationError) as caught:
        await verifier.verify("token")
    assert caught.value.code == "invalid_token"


@pytest.mark.asyncio
async def test_cloudflare_verifier_header_and_allowlist_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError):
        CloudflareAccessVerifier("", (), _StaticKeyCache())  # type: ignore[arg-type]
    verifier = CloudflareAccessVerifier(
        "https://team.cloudflareaccess.com",
        ("audience",),
        _StaticKeyCache(),  # type: ignore[arg-type]
        allowed_emails=frozenset({"owner@example.com"}),
    )
    for token in ("", "x" * (auth.MAX_JWT_BYTES + 1), None):
        with pytest.raises(AuthenticationError):
            await verifier.verify(token)  # type: ignore[arg-type]

    monkeypatch.setattr(
        jwt, "get_unverified_header", lambda token: {"alg": "HS256", "kid": "k"}
    )
    with pytest.raises(AuthenticationError):
        await verifier.verify("token")
    monkeypatch.setattr(jwt, "get_unverified_header", lambda token: {"alg": "RS256"})
    with pytest.raises(AuthenticationError):
        await verifier.verify("token")
    monkeypatch.setattr(
        jwt, "get_unverified_header", lambda token: {"alg": "RS256", "kid": "k"}
    )
    monkeypatch.setattr(
        jwt, "decode", lambda *args, **kwargs: {"exp": 1, "sub": "user"}
    )
    with pytest.raises(AuthenticationError) as caught:
        await verifier.verify("token")
    assert caught.value.code == "identity_not_allowed"


def _trusted_settings(tmp_path: Path) -> RuntimeSettings:
    return RuntimeSettings(
        enabled=True,
        public_origin="https://comfy.example.com",
        auth_mode=AuthMode.TRUSTED_PROXY,
        models_root=tmp_path / "models",
        state_directory=tmp_path / "state",
        trusted_proxy_networks=(ipaddress.ip_network("127.0.0.0/8"),),
    )


@pytest.mark.asyncio
async def test_authenticator_covers_fail_closed_modes_and_request_peer(
    tmp_path: Path,
) -> None:
    incomplete = RuntimeSettings(
        enabled=True,
        public_origin="https://comfy.example.com",
        auth_mode=AuthMode.CLOUDFLARE_ACCESS,
        models_root=tmp_path / "models",
        state_directory=tmp_path / "state",
    )
    with pytest.raises(ValueError, match="incomplete"):
        Authenticator(incomplete)

    unconfigured = RuntimeSettings(
        enabled=True,
        public_origin="https://comfy.example.com",
        auth_mode=None,
        models_root=tmp_path / "models",
        state_directory=tmp_path / "state",
    )
    with pytest.raises(AuthenticationUnavailable):
        await Authenticator(unconfigured).authenticate({}, "127.0.0.1")

    authenticator = Authenticator(_trusted_settings(tmp_path))
    transport = SimpleNamespace(get_extra_info=lambda name: ("127.0.0.2", 1234))
    request = SimpleNamespace(
        remote=None,
        transport=transport,
        headers={"X-Forwarded-User": "opaque-user"},
    )
    identity = await authenticator.authenticate_request(request)
    assert identity.subject == "trusted-proxy:opaque-user"
    assert identity.email is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("remote", "identity", "code"),
    [
        (None, "user", "untrusted_proxy"),
        ("not-an-ip", "user", "untrusted_proxy"),
        ("127.0.0.1", None, "authentication_required"),
        ("127.0.0.1", "", "invalid_identity"),
        ("127.0.0.1", "one,two", "invalid_identity"),
        ("127.0.0.1", "bad\nidentity", "invalid_identity"),
        ("127.0.0.1", "x" * 1025, "invalid_identity"),
    ],
)
async def test_trusted_proxy_rejects_ambiguous_peer_or_identity(
    tmp_path: Path, remote: str | None, identity: str | None, code: str
) -> None:
    headers = {} if identity is None else {"X-Forwarded-User": identity}
    with pytest.raises(AuthenticationError) as caught:
        await Authenticator(_trusted_settings(tmp_path)).authenticate(headers, remote)
    assert caught.value.code == code


@pytest.mark.asyncio
async def test_trusted_proxy_accepts_ipv4_mapped_loopback(tmp_path: Path) -> None:
    identity = await Authenticator(_trusted_settings(tmp_path)).authenticate(
        {"X-Forwarded-User": "user"}, "::ffff:127.0.0.3"
    )
    assert identity.subject == "trusted-proxy:user"


def test_settings_edge_validation_and_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SettingsError, match="boolean"):
        RuntimeSettings.from_env(
            {"SMD_ENABLED": "sometimes"}, folder_paths_module=object()
        )
    for origin in (None, "", "x" * 2049, "https://", "https://host:invalid"):
        with pytest.raises(SettingsError):
            settings.normalize_origin(origin)  # type: ignore[arg-type]
    assert settings.normalize_origin("https://[2001:db8::1]") == "https://[2001:db8::1]"
    assert RuntimeSettings(False, None, None, None, None).cf_jwks_url is None

    with pytest.raises(SettingsError, match="required"):
        settings._normalize_team_domain("  ")
    with pytest.raises(SettingsError, match="empty"):
        settings._split_values("one,,two", name="VALUES")
    for value in ("", "relative/path", " /tmp/path"):
        with pytest.raises(SettingsError):
            settings._parse_absolute_path(value, name="PATH")

    module = SimpleNamespace(
        models_dir=str(tmp_path / "models"),
        get_system_user_directory=lambda name: str(tmp_path / name),
    )
    monkeypatch.setattr(settings, "_import_folder_paths", lambda: module)
    assert settings.resolve_comfy_paths({}) == (
        tmp_path / "models",
        tmp_path / "server_model_downloader",
    )


def test_settings_reject_bad_comfy_resolvers_and_identity_fields(
    tmp_path: Path,
) -> None:
    class RaisingResolver:
        models_dir = tmp_path / "models"

        @staticmethod
        def get_system_user_directory(name: str) -> str:
            del name
            raise OSError("unavailable")

    with pytest.raises(SettingsError, match="could not resolve"):
        settings.resolve_comfy_paths({}, RaisingResolver())

    invalid_path_module = SimpleNamespace(
        models_dir=tmp_path / "models",
        get_system_user_directory=lambda name: object(),
    )
    with pytest.raises(SettingsError, match="did not return"):
        settings.resolve_comfy_paths({}, invalid_path_module)

    base = {
        "SMD_ENABLED": "false",
        "SMD_MODELS_ROOT": str(tmp_path / "models"),
        "SMD_STATE_DIR": str(tmp_path / "state"),
    }
    bad_updates = [
        {"SMD_ALLOWED_EMAILS": "not-an-email"},
        {"SMD_TRUSTED_IDENTITY_HEADER": "bad header"},
        {
            "SMD_AUTH_MODE": "cloudflare-access",
            "SMD_CF_TEAM_DOMAIN": "team.cloudflareaccess.com",
            "SMD_CF_AUDIENCE": "bad audience",
        },
    ]
    for update in bad_updates:
        with pytest.raises(SettingsError):
            RuntimeSettings.from_env(base | update)


def test_enabled_settings_report_all_missing_paths() -> None:
    with pytest.raises(SettingsError) as caught:
        RuntimeSettings.from_env(
            {
                "SMD_ENABLED": "true",
                "SMD_PUBLIC_ORIGIN": "https://comfy.example.com",
                "SMD_AUTH_MODE": "trusted-proxy",
            },
            folder_paths_module=object(),
        )
    assert "models path" in str(caught.value)
    assert "user path" in str(caught.value)


def test_safe_settings_loader_catches_unexpected_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("sensitive detail")

    monkeypatch.setattr(RuntimeSettings, "from_env", explode)
    state = settings.load_runtime_settings({})
    assert state.settings is None
    assert state.error == "runtime configuration could not be resolved"


def _signed_raw_payload(signer: TokenSigner, raw: bytes) -> str:
    body = security._b64encode(raw)
    signature = security._b64encode(
        hmac.new(signer._key, body.encode("ascii"), hashlib.sha256).digest()
    )
    return f"{body}.{signature}"


def test_token_signer_rejects_key_and_payload_edge_cases(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "download-token.key").write_bytes(b"short")
    with pytest.raises(SecurityError, match="invalid token key"):
        TokenSigner(state)

    signer = TokenSigner(tmp_path / "valid-state")
    assert TokenSigner(tmp_path / "valid-state")._key == signer._key
    with pytest.raises(SecurityError, match="too large"):
        signer.sign({"large": "x" * 20_000})
    with pytest.raises(SecurityError, match="malformed token payload"):
        signer.verify(_signed_raw_payload(signer, b"not-json"))
    for payload in (
        [1, 2],
        {"v": 2, "exp": int(time.time()) + 60},
        {"v": 1, "exp": True},
    ):
        raw = json.dumps(payload).encode("utf-8")
        with pytest.raises(SecurityError):
            signer.verify(_signed_raw_payload(signer, raw))


def test_bound_tokens_and_csrf_reject_invalid_binding_inputs(tmp_path: Path) -> None:
    signer = TokenSigner(tmp_path / "state")
    for subject in ("", "bad\nsubject", "x" * 2049, None):
        with pytest.raises(SecurityError, match="subject"):
            signer.sign_bound({}, subject)  # type: ignore[arg-type]
    with pytest.raises(SecurityError, match="purpose"):
        signer.sign_bound({}, "user", purpose="Invalid")
    with pytest.raises(SecurityError, match="reserved"):
        signer.sign_bound({"exp": 1}, "user")
    with pytest.raises(SecurityError):
        signer.issue_csrf("user", "http://comfy.example.com")
    token, _ = signer.issue_csrf("user", "https://comfy.example.com")
    with pytest.raises(SecurityError):
        signer.verify_csrf(token, "user", "http://comfy.example.com")
    with pytest.raises(SecurityError, match="origin"):
        signer.verify_csrf(token, "user", "https://other.example.com")


def test_origin_guards_reject_missing_host_origin_and_cross_site(
    tmp_path: Path,
) -> None:
    with pytest.raises(SecurityError, match="Host"):
        security.require_same_origin_and_csrf(
            {"X-SMD-CSRF": "csrf", "Origin": "https://comfy.example.com"}, "csrf"
        )

    signer = TokenSigner(tmp_path / "state")
    token, _ = signer.issue_csrf("user", "https://comfy.example.com")
    invalid_headers = [
        {"X-SMD-CSRF": token},
        {"X-SMD-CSRF": token, "Origin": "https://host:invalid"},
        {
            "X-SMD-CSRF": token,
            "Origin": "https://comfy.example.com",
            "Sec-Fetch-Site": "cross-site",
        },
        {"X-SMD-CSRF": "invalid", "Origin": "https://comfy.example.com"},
    ]
    for headers in invalid_headers:
        with pytest.raises(SecurityError):
            security.require_subject_origin_and_csrf(
                headers, signer, "user", "https://comfy.example.com"
            )


def test_api_fallback_response_and_json_safe_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "_web", None)
    response = api._json_response({"ok": True}, status=201)
    assert response.status == 201
    assert response.data == {"ok": True}
    assert response.headers["Content-Type"].startswith("application/json")

    class State(enum.Enum):
        READY = "ready"

    moment = dt.datetime(2026, 1, 2, 3, 4, tzinfo=dt.timezone.utc)
    assert api._json_safe(State.READY) == "ready"
    assert api._json_safe(Path("relative")) == "relative"
    assert api._json_safe(moment) == "2026-01-02T03:04:00Z"
    assert api._json_safe(dt.date(2026, 1, 2)) == "2026-01-02"
    assert api._json_safe({1: (Path("x"),)}) == {"1": ["x"]}
    with pytest.raises(TypeError, match="unsupported"):
        api._json_safe(object())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised", "status", "code"),
    [
        (api.APIError(409, "conflict", "conflict"), 409, "conflict"),
        (AuthenticationError("no", code="auth", status=401), 401, "auth"),
        (KeyError("missing"), 404, "not_found"),
        (SecurityError("unsafe"), 400, "invalid_request"),
        (ValueError("detail"), 400, "invalid_request"),
        (RuntimeError("detail"), 500, "internal_error"),
    ],
)
async def test_api_endpoint_maps_failure_families_to_stable_envelopes(
    monkeypatch: pytest.MonkeyPatch,
    raised: BaseException,
    status: int,
    code: str,
) -> None:
    monkeypatch.setattr(api, "_web", None)

    @api._endpoint
    async def handler(instance: object, request: object) -> None:
        del instance, request
        raise raised

    response = await handler(object(), object())
    assert response.status == status
    assert response.data["code"] == code
    assert response.data["request_id"]


class _BodyStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def iter_chunked(self, size: int):  # noqa: ANN201
        del size
        for chunk in self.chunks:
            yield chunk


@pytest.mark.asyncio
async def test_api_json_body_reader_covers_stream_read_and_size_errors() -> None:
    streamed = SimpleNamespace(
        headers={"Content-Type": "application/problem+json"},
        content_length=None,
        content=_BodyStream([b'{"ok":', b" true}"]),
    )
    assert await api._read_json_body(streamed) == {"ok": True}

    for length in ("bad", -1, api.MAX_BODY_BYTES + 1):
        request = SimpleNamespace(
            headers={"Content-Type": "application/json"},
            content_length=length,
            content=_BodyStream([b"{}"]),
        )
        with pytest.raises(api.APIError):
            await api._read_json_body(request)

    oversized = SimpleNamespace(
        headers={"Content-Type": "application/json"},
        content_length=None,
        content=_BodyStream([b"x" * (api.MAX_BODY_BYTES + 1)]),
    )
    with pytest.raises(api.APIError, match="exceeds"):
        await api._read_json_body(oversized)

    class ReadRequest:
        headers = {"Content-Type": "application/json"}
        content_length = None
        content = None

        def __init__(self, body: bytes) -> None:
            self.body = body

        async def read(self) -> bytes:
            return self.body

    assert await api._read_json_body(ReadRequest(b"[1]")) == [1]
    for body in (b"", b"not-json", b"\xff"):
        with pytest.raises(api.APIError):
            await api._read_json_body(ReadRequest(body))
    with pytest.raises(api.APIError, match="unavailable"):
        await api._read_json_body(
            SimpleNamespace(
                headers={"Content-Type": "application/json"},
                content_length=None,
                content=None,
            )
        )


@dataclasses.dataclass
class _PublicJob:
    id: str
    filename: str
    created_at: float


def test_api_job_projection_handles_supported_record_shapes() -> None:
    projected = api._job_to_dict(_PublicJob("one", "one.safetensors", 0))
    assert projected["name"] == "one.safetensors"
    assert projected["created_at"] == "1970-01-01T00:00:00Z"

    method_job = SimpleNamespace(
        public_dict=lambda: {
            "id": "two",
            "name": "Two",
            "completed_at": True,
            "secret": "hidden",
        }
    )
    assert api._job_to_dict(method_job) == {
        "id": "two",
        "name": "Two",
        "completed_at": True,
    }
    assert api._jobs_to_list({"jobs": None}) == []
    assert api._jobs_to_list({"id": "three"}) == [{"id": "three"}]
    with pytest.raises(TypeError, match="invalid job record"):
        api._job_to_dict(SimpleNamespace(public_dict=lambda: object()))
    with pytest.raises(TypeError, match="without an id"):
        api._job_to_dict({"status": "queued"})


@pytest.mark.asyncio
async def test_api_rate_limiter_prunes_stale_and_caps_identity_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter = api._InspectionRateLimiter(limit=2, window_seconds=10)
    monkeypatch.setattr(api.time, "monotonic", lambda: 100.0)
    limiter._events = {f"user-{index}": deque([0.0]) for index in range(1025)}
    await limiter.require_capacity("active")
    assert list(limiter._events) == ["active"]
    await limiter.require_capacity("active")
    with pytest.raises(api.APIError) as caught:
        await limiter.require_capacity("active")
    assert caught.value.code == "rate_limited"


@pytest.mark.asyncio
async def test_api_runtime_fail_closed_health_and_lifecycle_edges(
    tmp_path: Path,
) -> None:
    bad = api._Runtime(settings.SettingsState(None, "bad configuration"))
    with pytest.raises(api.APIError, match="invalid"):
        bad.require_configured()
    assert bad.health()[1] == 503
    with pytest.raises(api.APIError):
        _ = bad.models_root
    with pytest.raises(api.APIError):
        _ = bad.state_directory

    disabled = api._Runtime(
        settings.SettingsState(RuntimeSettings(False, None, None, None, None))
    )
    with pytest.raises(api.APIError, match="disabled"):
        disabled.require_configured()
    assert disabled.health()[0]["state"] == "disabled"
    await disabled.startup()
    await disabled.shutdown()

    configured = _trusted_settings(tmp_path)

    class Manager:
        async def ensure_started(self) -> None:
            return None

        def health(self) -> dict[str, Any]:
            return {"status": "ok", "state": "ready", "queued_jobs": 0}

        async def stop(self) -> None:
            return None

    runtime = api._Runtime(settings.SettingsState(configured), job_manager=Manager())
    assert await runtime.job_manager() is runtime._job_manager
    assert runtime.health() == (
        {"status": "ok", "state": "ready", "queued_jobs": 0},
        200,
    )
    await runtime.shutdown()


def test_job_store_migration_quarantine_pruning_and_cursors(tmp_path: Path) -> None:
    with pytest.raises(JobStoreError, match="unsupported"):
        migrate_job({}, 99)
    legacy = migrate_job({"id": "legacy", "status": "failed"}, 1)
    assert legacy["error_code"] == "legacy_failure"

    store = PersistentJobStore(tmp_path)
    store.path.write_text("[]", encoding="utf-8")
    assert store.load(lambda row: None) == ({}, True)
    assert store.last_quarantine is not None

    now = time.time()
    jobs = {
        "old": {
            "id": "old",
            "status": "completed",
            "created_at": 1,
            "completed_at": now - MAX_HISTORY_AGE_SECONDS - 1,
        },
        "active": {"id": "active", "status": "queued", "created_at": 0},
    }
    assert store.prune(jobs, now=now) is True
    assert set(jobs) == {"active"}

    oversized = {
        str(index): {
            "id": str(index),
            "status": "completed",
            "created_at": index,
            "completed_at": now,
        }
        for index in range(MAX_TERMINAL_HISTORY + 1)
    }
    assert store.prune(oversized, now=now) is True
    assert len(oversized) == MAX_TERMINAL_HISTORY

    cursor = encode_cursor({"id": "job", "created_at": 1})
    assert decode_cursor(cursor) == (1.0, "job")
    invalid_values = ("", "x" * 513, base64.urlsafe_b64encode(b"{}").decode())
    for value in invalid_values:
        with pytest.raises(JobStoreError):
            decode_cursor(value)


def test_state_directory_lease_release_is_idempotent(tmp_path: Path) -> None:
    lease = StateDirectoryLease(tmp_path)
    assert lease.acquired is True
    lease.acquire()
    lease.release()
    lease.release()
    assert lease.acquired is False
