from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from backend.auth import (
    AuthenticationError,
    AuthenticationUnavailable,
    Authenticator,
    CloudflareJWKSCache,
)
from backend.settings import AuthMode, RuntimeSettings


def _b64int(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _keypair(kid: str) -> tuple[Any, dict[str, str]]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    numbers = private_key.public_key().public_numbers()
    return private_key, {
        "kty": "RSA",
        "use": "sig",
        "alg": "RS256",
        "kid": kid,
        "n": _b64int(numbers.n),
        "e": _b64int(numbers.e),
    }


def _settings(tmp_path: Path, **extra: str) -> RuntimeSettings:
    env = {
        "SMD_ENABLED": "true",
        "SMD_PUBLIC_ORIGIN": "https://comfy.example.com",
        "SMD_AUTH_MODE": "cloudflare-access",
        "SMD_CF_TEAM_DOMAIN": "my-team.cloudflareaccess.com",
        "SMD_CF_AUDIENCE": "app-audience",
        "SMD_MODELS_ROOT": str(tmp_path / "models"),
        "SMD_STATE_DIR": str(tmp_path / "state"),
    }
    env.update(extra)
    return RuntimeSettings.from_env(env)


def _token(
    private_key: Any,
    *,
    kid: str = "key-1",
    issuer: str = "https://my-team.cloudflareaccess.com",
    audience: str = "app-audience",
    subject: str = "user-id",
    email: str = "owner@example.com",
    expires_delta: int = 300,
    not_before_delta: int = -1,
) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "iss": issuer,
            "aud": audience,
            "sub": subject,
            "email": email,
            "iat": now - 1,
            "nbf": now + not_before_delta,
            "exp": now + expires_delta,
        },
        private_key,
        algorithm="RS256",
        headers={"kid": kid},
    )


@pytest.mark.asyncio
async def test_cloudflare_access_verifies_rs256_claims_and_caches_jwks(
    tmp_path: Path,
) -> None:
    private_key, public_jwk = _keypair("key-1")
    calls = 0

    async def fetch(url: str) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        assert url.endswith("/cdn-cgi/access/certs")
        return {"keys": [public_jwk]}

    settings = _settings(tmp_path, SMD_ALLOWED_EMAILS="OWNER@example.com")
    cache = CloudflareJWKSCache(settings.cf_jwks_url or "", fetcher=fetch)
    authenticator = Authenticator(settings, jwks_cache=cache)
    token = _token(private_key)
    first = await authenticator.authenticate(
        {"Cf-Access-Jwt-Assertion": token}, "127.0.0.1"
    )
    second = await authenticator.authenticate(
        {"cf-access-jwt-assertion": token}, "10.0.0.1"
    )
    assert first == second
    assert first.subject == "cloudflare-access:user-id"
    assert first.email == "owner@example.com"
    assert first.auth_mode is AuthMode.CLOUDFLARE_ACCESS
    assert first.as_public_dict() == {
        "email": "owner@example.com",
        "auth_mode": "cloudflare-access",
    }
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token_kwargs",
    [
        {"expires_delta": -1},
        {"not_before_delta": 300},
        {"issuer": "https://other.cloudflareaccess.com"},
        {"audience": "wrong-audience"},
    ],
)
async def test_cloudflare_access_rejects_invalid_registered_claims(
    tmp_path: Path, token_kwargs: dict[str, Any]
) -> None:
    private_key, public_jwk = _keypair("key-1")

    async def fetch(url: str) -> dict[str, Any]:
        return {"keys": [public_jwk]}

    settings = _settings(tmp_path)
    authenticator = Authenticator(
        settings,
        jwks_cache=CloudflareJWKSCache(settings.cf_jwks_url or "", fetcher=fetch),
    )
    with pytest.raises(AuthenticationError) as error:
        await authenticator.authenticate(
            {"Cf-Access-Jwt-Assertion": _token(private_key, **token_kwargs)},
            "127.0.0.1",
        )
    assert error.value.code == "invalid_token"
    assert error.value.status == 401


@pytest.mark.asyncio
async def test_cloudflare_email_allowlist_is_fail_closed(tmp_path: Path) -> None:
    private_key, public_jwk = _keypair("key-1")

    async def fetch(url: str) -> dict[str, Any]:
        return {"keys": [public_jwk]}

    settings = _settings(tmp_path, SMD_ALLOWED_EMAILS="allowed@example.com")
    authenticator = Authenticator(
        settings,
        jwks_cache=CloudflareJWKSCache(settings.cf_jwks_url or "", fetcher=fetch),
    )
    with pytest.raises(AuthenticationError) as error:
        await authenticator.authenticate(
            {"Cf-Access-Jwt-Assertion": _token(private_key)}, "127.0.0.1"
        )
    assert error.value.code == "identity_not_allowed"
    assert error.value.status == 403


@pytest.mark.asyncio
async def test_jwks_rotation_refreshes_unknown_kid_and_outage_fails_closed(
    tmp_path: Path,
) -> None:
    first_private, first_public = _keypair("key-1")
    second_private, second_public = _keypair("key-2")
    clock = [100.0]
    responses: list[Any] = [
        {"keys": [first_public]},
        {"keys": [first_public, second_public]},
        AuthenticationUnavailable(),
    ]

    async def fetch(url: str) -> dict[str, Any]:
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    settings = _settings(tmp_path)
    cache = CloudflareJWKSCache(
        settings.cf_jwks_url or "",
        fetcher=fetch,
        ttl_seconds=10,
        min_refresh_interval_seconds=5,
        monotonic=lambda: clock[0],
    )
    authenticator = Authenticator(settings, jwks_cache=cache)
    await authenticator.authenticate(
        {"Cf-Access-Jwt-Assertion": _token(first_private, kid="key-1")},
        "127.0.0.1",
    )
    clock[0] += 6
    rotated = await authenticator.authenticate(
        {"Cf-Access-Jwt-Assertion": _token(second_private, kid="key-2")},
        "127.0.0.1",
    )
    assert rotated.subject == "cloudflare-access:user-id"
    clock[0] += 11
    with pytest.raises(AuthenticationUnavailable):
        await authenticator.authenticate(
            {"Cf-Access-Jwt-Assertion": _token(second_private, kid="key-2")},
            "127.0.0.1",
        )


@pytest.mark.asyncio
async def test_trusted_proxy_uses_socket_peer_and_configured_header(
    tmp_path: Path,
) -> None:
    env = {
        "SMD_ENABLED": "true",
        "SMD_PUBLIC_ORIGIN": "https://comfy.example.com",
        "SMD_AUTH_MODE": "trusted-proxy",
        "SMD_TRUSTED_PROXY_CIDRS": "10.10.0.0/16",
        "SMD_TRUSTED_IDENTITY_HEADER": "X-Authenticated-User",
        "SMD_MODELS_ROOT": str(tmp_path / "models"),
        "SMD_STATE_DIR": str(tmp_path / "state"),
    }
    authenticator = Authenticator(RuntimeSettings.from_env(env))
    identity = await authenticator.authenticate(
        {"X-Authenticated-User": "OWNER@Example.com"}, "10.10.3.4"
    )
    assert identity.subject == "trusted-proxy:OWNER@Example.com"
    assert identity.email == "owner@example.com"
    with pytest.raises(AuthenticationError) as error:
        await authenticator.authenticate(
            {"X-Authenticated-User": "owner@example.com"}, "10.11.3.4"
        )
    assert error.value.code == "untrusted_proxy"


@pytest.mark.asyncio
async def test_trusted_proxy_subject_does_not_casefold_distinct_identities(
    tmp_path: Path,
) -> None:
    settings = RuntimeSettings.from_env(
        {
            "SMD_ENABLED": "true",
            "SMD_PUBLIC_ORIGIN": "https://comfy.example.com",
            "SMD_AUTH_MODE": "trusted-proxy",
            "SMD_MODELS_ROOT": str(tmp_path / "models"),
            "SMD_STATE_DIR": str(tmp_path / "state"),
        }
    )
    authenticator = Authenticator(settings)
    first = await authenticator.authenticate(
        {"X-Forwarded-User": "CaseSensitiveID"}, "127.0.0.1"
    )
    second = await authenticator.authenticate(
        {"X-Forwarded-User": "casesensitiveid"}, "127.0.0.1"
    )
    assert first.subject != second.subject


@pytest.mark.asyncio
async def test_disabled_authenticator_is_unavailable() -> None:
    authenticator = Authenticator(
        RuntimeSettings.from_env({}, folder_paths_module=object())
    )
    with pytest.raises(AuthenticationUnavailable) as error:
        await authenticator.authenticate({}, "127.0.0.1")
    assert error.value.status == 503
