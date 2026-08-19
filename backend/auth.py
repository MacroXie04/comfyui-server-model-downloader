from __future__ import annotations

import asyncio
import ipaddress
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .settings import AuthMode, RuntimeSettings

JWT_HEADER = "Cf-Access-Jwt-Assertion"
MAX_JWT_BYTES = 32 * 1024
MAX_JWKS_BYTES = 1024 * 1024


class AuthenticationError(ValueError):
    """A browser-safe authentication failure with a stable API error code."""

    def __init__(
        self,
        message: str = "authentication required",
        *,
        code: str = "authentication_required",
        status: int = 401,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class AuthenticationUnavailable(AuthenticationError):
    def __init__(self, message: str = "authentication service is unavailable") -> None:
        super().__init__(message, code="authentication_unavailable", status=503)


@dataclass(frozen=True)
class Identity:
    subject: str
    email: str | None
    auth_mode: AuthMode

    def as_public_dict(self) -> dict[str, str | None]:
        return {"email": self.email, "auth_mode": self.auth_mode.value}


JWKSFetcher = Callable[[str], Awaitable[Mapping[str, Any]]]


def _header(headers: Mapping[str, str], name: str) -> str | None:
    direct = headers.get(name)
    if direct is not None:
        return str(direct)
    folded_name = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == folded_name:
            return str(value)
    return None


async def _default_jwks_fetcher(url: str) -> Mapping[str, Any]:
    try:
        import aiohttp
    except ImportError as exc:  # pragma: no cover - declared runtime dependency.
        raise AuthenticationUnavailable() from exc

    timeout = aiohttp.ClientTimeout(total=10, connect=5, sock_connect=5, sock_read=5)
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout, trust_env=False) as session,
            session.get(
                url,
                headers={"Accept": "application/json", "Accept-Encoding": "identity"},
                allow_redirects=False,
            ) as response,
        ):
            if response.status != 200:
                raise AuthenticationUnavailable()
            length = response.headers.get("Content-Length")
            if length:
                try:
                    if int(length) > MAX_JWKS_BYTES:
                        raise AuthenticationUnavailable()
                except ValueError as exc:
                    raise AuthenticationUnavailable() from exc
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.content.iter_chunked(64 * 1024):
                total += len(chunk)
                if total > MAX_JWKS_BYTES:
                    raise AuthenticationUnavailable()
                chunks.append(chunk)
    except AuthenticationError:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        raise AuthenticationUnavailable() from exc
    try:
        document = json.loads(b"".join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthenticationUnavailable() from exc
    if not isinstance(document, Mapping):
        raise AuthenticationUnavailable()
    return document


class CloudflareJWKSCache:
    """Bounded JWKS cache with refresh-on-rotation and fail-closed expiry."""

    def __init__(
        self,
        jwks_url: str,
        *,
        fetcher: JWKSFetcher | None = None,
        ttl_seconds: int = 3600,
        min_refresh_interval_seconds: int = 5,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not jwks_url.startswith("https://"):
            raise ValueError("JWKS URL must use HTTPS")
        if ttl_seconds < 1:
            raise ValueError("JWKS cache TTL must be positive")
        self.jwks_url = jwks_url
        self.fetcher = fetcher or _default_jwks_fetcher
        self.ttl_seconds = int(ttl_seconds)
        self.min_refresh_interval_seconds = max(0, int(min_refresh_interval_seconds))
        self.monotonic = monotonic
        self._keys: dict[str, Any] = {}
        self._expires_at = 0.0
        self._last_refresh = float("-inf")
        self._generation = 0
        self._lock: asyncio.Lock | None = None

    def _lock_for_loop(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @staticmethod
    def _parse_keys(document: Mapping[str, Any]) -> dict[str, Any]:
        try:
            import jwt
        except ImportError as exc:  # pragma: no cover - declared runtime dependency.
            raise AuthenticationUnavailable() from exc

        rows = document.get("keys")
        if not isinstance(rows, list) or not rows:
            raise AuthenticationUnavailable()
        parsed: dict[str, Any] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise AuthenticationUnavailable()
            kid = row.get("kid")
            if not isinstance(kid, str) or not kid or len(kid) > 512:
                raise AuthenticationUnavailable()
            if kid in parsed:
                raise AuthenticationUnavailable()
            if row.get("kty") != "RSA" or row.get("alg", "RS256") != "RS256":
                continue
            if row.get("use", "sig") != "sig":
                continue
            try:
                parsed[kid] = jwt.PyJWK.from_dict(dict(row), algorithm="RS256").key
            except (TypeError, ValueError, jwt.PyJWTError) as exc:
                raise AuthenticationUnavailable() from exc
        if not parsed:
            raise AuthenticationUnavailable()
        return parsed

    async def _refresh(self) -> None:
        try:
            document = await self.fetcher(self.jwks_url)
            keys = self._parse_keys(document)
        except AuthenticationError:
            raise
        except Exception as exc:
            raise AuthenticationUnavailable() from exc
        now = self.monotonic()
        self._keys = keys
        self._last_refresh = now
        self._expires_at = now + self.ttl_seconds
        self._generation += 1

    async def get_key(self, kid: str) -> Any:
        if not isinstance(kid, str) or not kid or len(kid) > 512:
            raise AuthenticationError("invalid access token", code="invalid_token")
        now = self.monotonic()
        key = self._keys.get(kid)
        if key is not None and now < self._expires_at:
            return key

        generation = self._generation
        async with self._lock_for_loop():
            now = self.monotonic()
            key = self._keys.get(kid)
            if key is not None and now < self._expires_at:
                return key
            if self._generation != generation and now < self._expires_at:
                key = self._keys.get(kid)
                if key is None:
                    raise AuthenticationError(
                        "invalid access token", code="invalid_token"
                    )
                return key
            unknown_in_fresh_cache = (
                bool(self._keys)
                and now < self._expires_at
                and now - self._last_refresh < self.min_refresh_interval_seconds
            )
            if not unknown_in_fresh_cache:
                await self._refresh()
            key = self._keys.get(kid)
            if key is None:
                raise AuthenticationError("invalid access token", code="invalid_token")
            return key


class CloudflareAccessVerifier:
    def __init__(
        self,
        issuer: str,
        audiences: tuple[str, ...],
        jwks_cache: CloudflareJWKSCache,
        *,
        allowed_emails: frozenset[str] = frozenset(),
    ) -> None:
        if not issuer or not audiences:
            raise ValueError("Cloudflare issuer and audience are required")
        self.issuer = issuer
        self.audiences = audiences
        self.jwks_cache = jwks_cache
        self.allowed_emails = allowed_emails

    async def verify(self, token: str) -> Identity:
        try:
            import jwt
        except ImportError as exc:  # pragma: no cover - declared runtime dependency.
            raise AuthenticationUnavailable() from exc

        if (
            not isinstance(token, str)
            or not token
            or len(token.encode("utf-8")) > MAX_JWT_BYTES
        ):
            raise AuthenticationError("invalid access token", code="invalid_token")
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise AuthenticationError(
                "invalid access token", code="invalid_token"
            ) from exc
        if header.get("alg") != "RS256":
            raise AuthenticationError("invalid access token", code="invalid_token")
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            raise AuthenticationError("invalid access token", code="invalid_token")
        key = await self.jwks_cache.get_key(kid)
        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=["RS256"],
                audience=list(self.audiences),
                issuer=self.issuer,
                options={
                    "require": ["exp", "iss", "aud", "sub"],
                    "verify_exp": True,
                    "verify_nbf": True,
                },
            )
        except jwt.PyJWTError as exc:
            raise AuthenticationError(
                "invalid access token", code="invalid_token"
            ) from exc
        if not isinstance(claims, Mapping):
            raise AuthenticationError("invalid access token", code="invalid_token")
        for temporal_claim in ("exp", "nbf"):
            value = claims.get(temporal_claim)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise AuthenticationError("invalid access token", code="invalid_token")
        subject = claims.get("sub")
        if (
            not isinstance(subject, str)
            or not subject
            or len(subject) > 1024
            or any(
                ord(character) < 32 or ord(character) == 127 for character in subject
            )
        ):
            raise AuthenticationError("invalid access token", code="invalid_token")
        raw_email = claims.get("email")
        email: str | None = None
        if raw_email is not None:
            if (
                not isinstance(raw_email, str)
                or not raw_email
                or len(raw_email) > 320
                or any(
                    ord(character) < 33 or ord(character) == 127
                    for character in raw_email
                )
            ):
                raise AuthenticationError("invalid access token", code="invalid_token")
            email = raw_email.casefold()
        if self.allowed_emails and (email is None or email not in self.allowed_emails):
            raise AuthenticationError(
                "access is not permitted for this identity",
                code="identity_not_allowed",
                status=403,
            )
        return Identity(
            subject=f"cloudflare-access:{subject}",
            email=email,
            auth_mode=AuthMode.CLOUDFLARE_ACCESS,
        )


class Authenticator:
    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        jwks_cache: CloudflareJWKSCache | None = None,
    ) -> None:
        self.settings = settings
        self._cloudflare: CloudflareAccessVerifier | None = None
        if settings.enabled and settings.auth_mode is AuthMode.CLOUDFLARE_ACCESS:
            if (
                not settings.cf_issuer
                or not settings.cf_jwks_url
                or not settings.cf_audiences
            ):
                raise ValueError("incomplete Cloudflare Access settings")
            cache = jwks_cache or CloudflareJWKSCache(settings.cf_jwks_url)
            self._cloudflare = CloudflareAccessVerifier(
                settings.cf_issuer,
                settings.cf_audiences,
                cache,
                allowed_emails=settings.allowed_emails,
            )

    async def authenticate(
        self, headers: Mapping[str, str], remote_address: str | None
    ) -> Identity:
        if not self.settings.enabled:
            raise AuthenticationUnavailable("server model downloader is disabled")
        if self.settings.auth_mode is AuthMode.CLOUDFLARE_ACCESS:
            token = _header(headers, JWT_HEADER)
            if not token:
                raise AuthenticationError()
            if self._cloudflare is None:
                raise AuthenticationUnavailable()
            return await self._cloudflare.verify(token)
        if self.settings.auth_mode is AuthMode.TRUSTED_PROXY:
            return self._authenticate_trusted_proxy(headers, remote_address)
        raise AuthenticationUnavailable(
            "server model downloader authentication is not configured"
        )

    async def authenticate_request(self, request: Any) -> Identity:
        remote = getattr(request, "remote", None)
        if not remote:
            transport = getattr(request, "transport", None)
            peername = (
                transport.get_extra_info("peername") if transport is not None else None
            )
            if isinstance(peername, tuple) and peername:
                remote = peername[0]
        return await self.authenticate(request.headers, str(remote) if remote else None)

    def _authenticate_trusted_proxy(
        self, headers: Mapping[str, str], remote_address: str | None
    ) -> Identity:
        if not remote_address:
            raise AuthenticationError(
                "untrusted proxy", code="untrusted_proxy", status=403
            )
        try:
            address = ipaddress.ip_address(remote_address)
        except ValueError as exc:
            raise AuthenticationError(
                "untrusted proxy", code="untrusted_proxy", status=403
            ) from exc
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        if not any(
            address in network for network in self.settings.trusted_proxy_networks
        ):
            raise AuthenticationError(
                "untrusted proxy", code="untrusted_proxy", status=403
            )
        raw_identity = _header(headers, self.settings.trusted_identity_header)
        if raw_identity is None:
            raise AuthenticationError()
        identity = raw_identity.strip()
        if (
            not identity
            or len(identity) > 1024
            or "," in identity
            or any(
                ord(character) < 32 or ord(character) == 127 for character in identity
            )
        ):
            raise AuthenticationError("invalid proxy identity", code="invalid_identity")
        email = identity.casefold() if "@" in identity else None
        return Identity(
            subject=f"trusted-proxy:{identity}",
            email=email,
            auth_mode=AuthMode.TRUSTED_PROXY,
        )


__all__ = [
    "JWT_HEADER",
    "AuthenticationError",
    "AuthenticationUnavailable",
    "Authenticator",
    "CloudflareAccessVerifier",
    "CloudflareJWKSCache",
    "Identity",
]
