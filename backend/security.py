from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import socket
import stat
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from .settings import SettingsError, normalize_origin

SAFE_EXTENSIONS = (".safetensors",)
ALLOWED_DIRECTORIES = (
    "checkpoints",
    "diffusion_models",
    "text_encoders",
    "clip",
    "clip_vision",
    "vae",
    "loras",
    "controlnet",
    "upscale_models",
    "embeddings",
    "audio_encoders",
    "style_models",
)
RESERVED_QUERY_KEYS = {
    "token",
    "api_key",
    "apikey",
    "access_token",
    "auth",
    "authorization",
    "key",
}
_HF_REDIRECT_SUFFIXES = (
    "huggingface.co",
    ".huggingface.co",
    "hf.co",
    ".hf.co",
    "xethub.hf.co",
    ".xethub.hf.co",
)
_CIVITAI_REDIRECT_SUFFIXES = (
    "civitai.com",
    ".civitai.com",
    "civitai.green",
    ".civitai.green",
    ".r2.cloudflarestorage.com",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_CIVITAI_DOWNLOAD_RE = re.compile(r"^/api/download/models/([0-9]+)/?$")


class SecurityError(ValueError):
    pass


@dataclass(frozen=True)
class SourceURL:
    provider: str
    url: str
    repo_id: str | None = None
    revision: str | None = None
    repo_path: str | None = None
    version_id: int | None = None
    file_id: int | None = None


def _validate_common_url(url: str) -> tuple[Any, str]:
    if not isinstance(url, str) or len(url) > 4096:
        raise SecurityError("URL must be a string no longer than 4096 characters")
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https":
        raise SecurityError("only HTTPS URLs are allowed")
    if parsed.username is not None or parsed.password is not None:
        raise SecurityError("URL credentials are not allowed")
    if parsed.fragment:
        raise SecurityError("URL fragments are not allowed")
    try:
        port = parsed.port
    except ValueError as exc:
        raise SecurityError("invalid URL port") from exc
    if port not in (None, 443):
        raise SecurityError("non-default ports are not allowed")
    host = (parsed.hostname or "").rstrip(".").lower()
    if not host:
        raise SecurityError("URL host is required")
    for key, _ in parse_qsl(parsed.query, keep_blank_values=True):
        if key.lower() in RESERVED_QUERY_KEYS:
            raise SecurityError("credentials in URL query strings are not allowed")
    return parsed, host


def _safe_segments(path: str) -> list[str]:
    raw_segments = path.split("/")
    segments: list[str] = []
    for raw in raw_segments:
        if raw == "":
            continue
        value = unquote(raw)
        if (
            value in (".", "..")
            or len(value) > 255
            or "\\" in value
            or "\x00" in value
            or "/" in value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise SecurityError("unsafe URL path segment")
        segments.append(value)
    return segments


def validate_source_url(url: str) -> SourceURL:
    parsed, host = _validate_common_url(url)
    if host == "huggingface.co":
        if parsed.query not in ("", "download=true"):
            raise SecurityError(
                "Hugging Face resolve URLs only allow the exact query download=true"
            )
        parts = _safe_segments(parsed.path)
        if len(parts) < 5 or parts[2] != "resolve":
            raise SecurityError(
                "Hugging Face URL must be /OWNER/REPO/resolve/REVISION/FILE"
            )
        repo_id = f"{parts[0]}/{parts[1]}"
        revision = parts[3]
        component_re = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
        if not component_re.fullmatch(parts[0]) or not component_re.fullmatch(parts[1]):
            raise SecurityError("unsafe Hugging Face repository identifier")
        if not component_re.fullmatch(revision):
            raise SecurityError("unsafe Hugging Face revision")
        repo_path = "/".join(parts[4:])
        validate_filename(parts[-1])
        return SourceURL("huggingface", url, repo_id, revision, repo_path, None, None)
    if host == "civitai.com":
        match = _CIVITAI_DOWNLOAD_RE.fullmatch(parsed.path)
        if not match:
            raise SecurityError(
                "Civitai URL must be https://civitai.com/api/download/models/ID"
            )
        # Non-secret selector parameters used by Civitai are permitted. Unknown
        # parameters are rejected so a signed token cannot become a URL smuggler.
        allowed_query = {"type", "format", "size", "fp", "fileId"}
        file_id = None
        seen_query_keys: set[str] = set()
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            if key not in allowed_query:
                raise SecurityError(f"unsupported Civitai query parameter: {key}")
            if key in seen_query_keys:
                raise SecurityError(f"duplicate Civitai query parameter: {key}")
            seen_query_keys.add(key)
            if key == "fileId":
                if not value.isdigit():
                    raise SecurityError("Civitai fileId must be numeric")
                file_id = int(value)
            elif not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", value):
                raise SecurityError(f"unsafe Civitai selector value: {key}")
        return SourceURL(
            "civitai", url, version_id=int(match.group(1)), file_id=file_id
        )
    raise SecurityError(
        "only huggingface.co resolve URLs and civitai.com download URLs are allowed"
    )


def validate_redirect_url(url: str, provider: str) -> str:
    _, host = _validate_common_url(url)
    if provider == "huggingface":
        suffixes = _HF_REDIRECT_SUFFIXES
    elif provider == "civitai":
        suffixes = _CIVITAI_REDIRECT_SUFFIXES
    else:
        raise SecurityError("unknown download provider")
    if not any(
        host == suffix.lstrip(".") or (suffix.startswith(".") and host.endswith(suffix))
        for suffix in suffixes
    ):
        raise SecurityError(f"redirect host is not an approved {provider} CDN")
    return host


def validate_filename(filename: str) -> str:
    if not isinstance(filename, str) or not filename or len(filename) > 255:
        raise SecurityError("invalid filename")
    if filename in (".", "..") or Path(filename).name != filename:
        raise SecurityError("filename must not contain a path")
    if (
        "\x00" in filename
        or "\\" in filename
        or ":" in filename
        or ".." in filename
        or filename.startswith(".")
        or any(ord(character) < 32 or ord(character) == 127 for character in filename)
    ):
        raise SecurityError("unsafe filename")
    if Path(filename).suffix.lower() not in SAFE_EXTENSIONS:
        raise SecurityError("only .safetensors files are allowed")
    return filename


def validate_directory(directory: str) -> str:
    if directory not in ALLOWED_DIRECTORIES:
        raise SecurityError("destination directory is not allowlisted")
    return directory


def validate_sha256(value: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise SecurityError("a valid SHA256 is required")
    return value.lower()


async def require_public_dns(host: str, port: int = 443) -> None:
    """Resolve a host and reject every non-public result.

    This is deliberately applied to each redirect hop as well as the initial
    provider endpoint. Official endpoints returning mixed public/private DNS
    are rejected rather than partially trusted.
    """

    import asyncio

    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SecurityError(f"DNS resolution failed for {host}") from exc
    if not infos:
        raise SecurityError(f"DNS returned no addresses for {host}")
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            not address.is_global
            or address.is_multicast
            or address.is_unspecified
            or address.is_loopback
            or address.is_link_local
            or address.is_private
            or address.is_reserved
        ):
            raise SecurityError(f"non-public address rejected for {host}")


def ensure_state_directory(path: Path) -> Path:
    path = Path(path).absolute()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    mode = os.lstat(path).st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise SecurityError("state directory must be a real directory")
    os.chmod(path, 0o700)
    return path.resolve(strict=True)


def resolve_model_paths(
    models_root: Path, directory: str, filename: str
) -> tuple[Path, Path]:
    directory = validate_directory(directory)
    filename = validate_filename(filename)
    root_input = Path(models_root).absolute()
    try:
        root_mode = os.lstat(root_input).st_mode
    except FileNotFoundError as exc:
        raise SecurityError("models root must already exist") from exc
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise SecurityError("models root must be a real directory")
    root = root_input.resolve(strict=True)
    destination_input = root / directory
    destination_input.mkdir(mode=0o750, exist_ok=True)
    destination_mode = os.lstat(destination_input).st_mode
    if stat.S_ISLNK(destination_mode) or not stat.S_ISDIR(destination_mode):
        raise SecurityError("destination directory may not be a symlink")
    destination = destination_input.resolve(strict=True)
    if destination.parent != root:
        raise SecurityError("destination escapes models root")
    final_path = destination / filename
    part_path = destination / f"{filename}.part"
    for candidate in (final_path, part_path):
        try:
            candidate_mode = os.lstat(candidate).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(candidate_mode) or not stat.S_ISREG(candidate_mode):
            raise SecurityError("model and partial files must be regular files")
    return final_path, part_path


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise SecurityError("malformed token")
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


class TokenSigner:
    def __init__(self, state_directory: Path):
        self.state_directory = ensure_state_directory(state_directory)
        self._key = self._load_or_create_key(
            self.state_directory / "download-token.key"
        )
        self.csrf_token = _b64encode(
            hmac.new(
                self._key, b"server-model-downloader-csrf-v1", hashlib.sha256
            ).digest()
        )

    @staticmethod
    def _load_or_create_key(path: Path) -> bytes:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(path, flags)
        except FileNotFoundError:
            create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                create_flags |= os.O_NOFOLLOW
            try:
                fd = os.open(path, create_flags, 0o600)
                key = secrets.token_bytes(32)
                os.write(fd, key)
                os.fsync(fd)
                os.close(fd)
                return key
            except FileExistsError:
                fd = os.open(path, flags)
        try:
            mode = os.fstat(fd).st_mode
            if not stat.S_ISREG(mode):
                raise SecurityError("token key must be a regular file")
            key = os.read(fd, 128)
        finally:
            os.close(fd)
        if len(key) != 32:
            raise SecurityError("invalid token key")
        os.chmod(path, 0o600)
        return key

    def sign(
        self, payload: Mapping[str, Any], ttl_seconds: int = 1800
    ) -> tuple[str, int]:
        expires_at = int(time.time()) + ttl_seconds
        data = dict(payload)
        data.update({"v": 1, "exp": expires_at})
        raw = json.dumps(
            data, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        if len(raw) > 16_384:
            raise SecurityError("token payload is too large")
        body = _b64encode(raw)
        signature = _b64encode(
            hmac.new(self._key, body.encode("ascii"), hashlib.sha256).digest()
        )
        return f"{body}.{signature}", expires_at

    def verify(self, token: str) -> dict[str, Any]:
        if not isinstance(token, str) or len(token) > 32_768 or token.count(".") != 1:
            raise SecurityError("malformed token")
        body, signature = token.split(".", 1)
        expected = _b64encode(
            hmac.new(self._key, body.encode("ascii"), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(signature, expected):
            raise SecurityError("invalid token signature")
        try:
            payload = json.loads(_b64decode(body))
        except (ValueError, json.JSONDecodeError) as exc:
            raise SecurityError("malformed token payload") from exc
        if not isinstance(payload, dict) or payload.get("v") != 1:
            raise SecurityError("unsupported token version")
        if not isinstance(payload.get("exp"), int) or payload["exp"] <= int(
            time.time()
        ):
            raise SecurityError("download token has expired")
        return payload

    @staticmethod
    def _validate_subject(subject: str) -> str:
        if (
            not isinstance(subject, str)
            or not subject
            or len(subject) > 2048
            or any(
                ord(character) < 32 or ord(character) == 127 for character in subject
            )
        ):
            raise SecurityError("invalid authenticated subject")
        return subject

    def _subject_binding(self, subject: str) -> str:
        subject = self._validate_subject(subject)
        return _b64encode(
            hmac.new(
                self._key,
                b"server-model-downloader-subject-v1\x00" + subject.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        )

    def subject_binding(self, subject: str) -> str:
        """Return the non-reversible binding used to scope persisted state."""

        return self._subject_binding(subject)

    def sign_bound(
        self,
        payload: Mapping[str, Any],
        subject: str,
        *,
        purpose: str = "download",
        ttl_seconds: int = 1800,
    ) -> tuple[str, int]:
        """Sign a token that can only be used by one authenticated subject."""

        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", purpose):
            raise SecurityError("invalid token purpose")
        data = dict(payload)
        reserved = {"_smd_subject", "_smd_purpose", "v", "exp"}
        if reserved.intersection(data):
            raise SecurityError("token payload contains reserved fields")
        data["_smd_subject"] = self._subject_binding(subject)
        data["_smd_purpose"] = purpose
        return self.sign(data, ttl_seconds=ttl_seconds)

    def verify_bound(
        self,
        token: str,
        subject: str,
        *,
        purpose: str = "download",
    ) -> dict[str, Any]:
        payload = self.verify(token)
        supplied_binding = payload.get("_smd_subject")
        if (
            not isinstance(supplied_binding, str)
            or not hmac.compare_digest(supplied_binding, self._subject_binding(subject))
            or payload.get("_smd_purpose") != purpose
        ):
            raise SecurityError("token is not valid for this identity")
        return payload

    def sign_download(
        self,
        payload: Mapping[str, Any],
        subject: str,
        *,
        ttl_seconds: int = 1800,
    ) -> tuple[str, int]:
        return self.sign_bound(
            payload, subject, purpose="download", ttl_seconds=ttl_seconds
        )

    def verify_download(self, token: str, subject: str) -> dict[str, Any]:
        return self.verify_bound(token, subject, purpose="download")

    def issue_csrf(
        self,
        subject: str,
        public_origin: str,
        *,
        ttl_seconds: int = 15 * 60,
    ) -> tuple[str, int]:
        try:
            origin = normalize_origin(public_origin, name="public origin")
        except SettingsError as exc:
            raise SecurityError(str(exc)) from exc
        return self.sign_bound(
            {"origin": origin},
            subject,
            purpose="csrf",
            ttl_seconds=ttl_seconds,
        )

    def verify_csrf(
        self, token: str, subject: str, public_origin: str
    ) -> dict[str, Any]:
        try:
            origin = normalize_origin(public_origin, name="public origin")
        except SettingsError as exc:
            raise SecurityError(str(exc)) from exc
        payload = self.verify_bound(token, subject, purpose="csrf")
        if payload.get("origin") != origin:
            raise SecurityError("CSRF token is not valid for this origin")
        return payload


def require_same_origin_and_csrf(
    headers: Mapping[str, str], expected_csrf: str
) -> None:
    supplied = headers.get("X-SMD-CSRF", "")
    if not supplied or not hmac.compare_digest(supplied, expected_csrf):
        raise SecurityError("missing or invalid CSRF token")
    origin = headers.get("Origin")
    if not origin:
        raise SecurityError("Origin header is required")
    parsed, origin_host = _validate_common_url(origin)
    if parsed.path not in ("", "/") or parsed.query:
        raise SecurityError("invalid Origin header")
    forwarded_host = headers.get("X-Forwarded-Host", "").split(",", 1)[0].strip()
    request_host = (
        (forwarded_host or headers.get("Host", "")).strip().lower().rstrip(".")
    )
    if not request_host:
        raise SecurityError("request Host header is required")
    request_host = request_host.removesuffix(":443")
    if origin_host != request_host:
        raise SecurityError("cross-origin request rejected")
    fetch_site = headers.get("Sec-Fetch-Site")
    if fetch_site and fetch_site not in ("same-origin", "none"):
        raise SecurityError("cross-site request rejected")


def require_subject_origin_and_csrf(
    headers: Mapping[str, str],
    signer: TokenSigner,
    subject: str,
    public_origin: str,
) -> dict[str, Any]:
    """Require an exact configured Origin and a short-lived subject-bound CSRF."""

    supplied = headers.get("X-SMD-CSRF", "")
    if not supplied:
        raise SecurityError("missing or invalid CSRF token")
    origin = headers.get("Origin")
    if not origin:
        raise SecurityError("Origin header is required")
    try:
        actual_origin = normalize_origin(origin, name="Origin header")
        expected_origin = normalize_origin(public_origin, name="public origin")
    except SettingsError as exc:
        raise SecurityError(str(exc)) from exc
    if actual_origin != expected_origin:
        raise SecurityError("cross-origin request rejected")
    fetch_site = headers.get("Sec-Fetch-Site")
    if fetch_site and fetch_site not in ("same-origin", "none"):
        raise SecurityError("cross-site request rejected")
    try:
        return signer.verify_csrf(supplied, subject, expected_origin)
    except SecurityError as exc:
        raise SecurityError("missing or invalid CSRF token") from exc
