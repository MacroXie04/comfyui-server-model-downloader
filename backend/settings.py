from __future__ import annotations

import importlib
import ipaddress
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from urllib.parse import urlsplit


class SettingsError(ValueError):
    """Raised when the downloader configuration is unsafe or incomplete."""


class AuthMode(str, Enum):
    CLOUDFLARE_ACCESS = "cloudflare-access"
    TRUSTED_PROXY = "trusted-proxy"


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_FORBIDDEN_IDENTITY_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "host",
        "origin",
        "x-smd-csrf",
        "cf-access-jwt-assertion",
    }
)
DEFAULT_TRUSTED_PROXY_CIDRS = ("127.0.0.0/8", "::1/128")
DEFAULT_TRUSTED_IDENTITY_HEADER = "X-Forwarded-User"


def _parse_bool(value: str | None, *, name: str, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise SettingsError(f"{name} must be a boolean")


def normalize_origin(value: str, *, name: str = "origin") -> str:
    """Return the canonical, root-only HTTPS origin used for exact matching."""

    if not isinstance(value, str) or not value or len(value) > 2048:
        raise SettingsError(f"{name} must be a non-empty HTTPS origin")
    if value != value.strip():
        raise SettingsError(f"{name} may not contain surrounding whitespace")
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise SettingsError(f"{name} is not a valid HTTPS origin") from exc
    if parsed.scheme.lower() != "https":
        raise SettingsError(f"{name} must use HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise SettingsError(f"{name} may not contain credentials")
    if not parsed.hostname:
        raise SettingsError(f"{name} must include a host")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise SettingsError(
            f"{name} must be an origin without a path, query, or fragment"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise SettingsError(f"{name} contains an invalid port") from exc
    if port not in (None, 443):
        raise SettingsError(f"{name} must use the default HTTPS port")
    raw_host = parsed.hostname
    if raw_host.endswith("."):
        raise SettingsError(f"{name} may not use a trailing-dot hostname")
    host = raw_host.lower()
    if not host or any(
        ord(character) < 33 or ord(character) == 127 for character in host
    ):
        raise SettingsError(f"{name} contains an invalid host")
    try:
        rendered_host = (
            f"[{ipaddress.ip_address(host).compressed}]" if ":" in host else host
        )
    except ValueError:
        rendered_host = host
    return f"https://{rendered_host}"


def _normalize_team_domain(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise SettingsError("SMD_CF_TEAM_DOMAIN is required")
    if "://" not in raw:
        raw = f"https://{raw}"
    issuer = normalize_origin(raw, name="SMD_CF_TEAM_DOMAIN")
    host = (urlsplit(issuer).hostname or "").lower()
    if not host.endswith(".cloudflareaccess.com") or host == "cloudflareaccess.com":
        raise SettingsError(
            "SMD_CF_TEAM_DOMAIN must be a tenant subdomain of cloudflareaccess.com"
        )
    return issuer


def _split_values(value: str | None, *, name: str) -> tuple[str, ...]:
    if value is None or not value.strip():
        return ()
    values = tuple(part.strip() for part in value.split(","))
    if any(not part for part in values):
        raise SettingsError(f"{name} contains an empty value")
    return tuple(dict.fromkeys(values))


def _parse_absolute_path(value: str, *, name: str) -> Path:
    if not value or value != value.strip() or "\x00" in value:
        raise SettingsError(f"{name} must be a non-empty absolute path")
    try:
        path = Path(value).expanduser()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SettingsError(f"{name} is not a valid filesystem path") from exc
    if not path.is_absolute():
        raise SettingsError(f"{name} must be an absolute path")
    return path


def _import_folder_paths() -> ModuleType | None:
    try:
        return importlib.import_module("folder_paths")
    except ImportError:
        return None


def resolve_comfy_paths(
    env: Mapping[str, str], folder_paths_module: object | None = None
) -> tuple[Path | None, Path | None]:
    """Resolve explicit overrides first and ComfyUI-native paths second.

    Returning ``None`` outside a ComfyUI process is intentional: a disabled
    extension can still be imported for packaging and tests, while enabling it
    without real ComfyUI paths fails closed in :meth:`RuntimeSettings.from_env`.
    """

    models_override = env.get("SMD_MODELS_ROOT")
    state_override = env.get("SMD_STATE_DIR")
    module = folder_paths_module
    if module is None and (models_override is None or state_override is None):
        module = _import_folder_paths()

    if models_override is not None:
        models_root = _parse_absolute_path(models_override, name="SMD_MODELS_ROOT")
    else:
        raw_models = getattr(module, "models_dir", None) if module is not None else None
        if raw_models is None:
            models_root = None
        else:
            try:
                rendered_models = os.fspath(raw_models)
            except TypeError as exc:
                raise SettingsError(
                    "folder_paths.models_dir is not a filesystem path"
                ) from exc
            models_root = _parse_absolute_path(
                rendered_models, name="folder_paths.models_dir"
            )

    if state_override is not None:
        state_directory = _parse_absolute_path(state_override, name="SMD_STATE_DIR")
    else:
        resolver = (
            getattr(module, "get_system_user_directory", None)
            if module is not None
            else None
        )
        if callable(resolver):
            try:
                raw_state = resolver("server_model_downloader")
            except (OSError, TypeError, ValueError) as exc:
                raise SettingsError(
                    "folder_paths.get_system_user_directory could not resolve "
                    "state storage"
                ) from exc
            try:
                rendered_state = os.fspath(raw_state)
            except TypeError as exc:
                raise SettingsError(
                    "folder_paths.get_system_user_directory did not return a path"
                ) from exc
            state_directory = _parse_absolute_path(
                rendered_state, name="folder_paths.get_system_user_directory"
            )
        else:
            state_directory = None
    return models_root, state_directory


@dataclass(frozen=True)
class RuntimeSettings:
    enabled: bool
    public_origin: str | None
    auth_mode: AuthMode | None
    models_root: Path | None
    state_directory: Path | None
    cf_issuer: str | None = None
    cf_audiences: tuple[str, ...] = ()
    allowed_emails: frozenset[str] = frozenset()
    trusted_proxy_networks: tuple[
        ipaddress.IPv4Network | ipaddress.IPv6Network, ...
    ] = ()
    trusted_identity_header: str = DEFAULT_TRUSTED_IDENTITY_HEADER

    @property
    def cf_jwks_url(self) -> str | None:
        if self.cf_issuer is None:
            return None
        return f"{self.cf_issuer}/cdn-cgi/access/certs"

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        folder_paths_module: object | None = None,
    ) -> RuntimeSettings:
        values = os.environ if env is None else env
        enabled = _parse_bool(
            values.get("SMD_ENABLED"), name="SMD_ENABLED", default=False
        )
        models_root, state_directory = resolve_comfy_paths(values, folder_paths_module)

        raw_origin = values.get("SMD_PUBLIC_ORIGIN")
        public_origin = (
            normalize_origin(raw_origin, name="SMD_PUBLIC_ORIGIN")
            if raw_origin is not None
            else None
        )

        raw_mode = values.get("SMD_AUTH_MODE")
        auth_mode: AuthMode | None = None
        if raw_mode is not None and raw_mode.strip():
            try:
                auth_mode = AuthMode(raw_mode.strip().lower())
            except ValueError as exc:
                raise SettingsError(
                    "SMD_AUTH_MODE must be cloudflare-access or trusted-proxy"
                ) from exc

        cf_issuer: str | None = None
        cf_audiences: tuple[str, ...] = ()
        allowed_emails = frozenset(
            email.casefold()
            for email in _split_values(
                values.get("SMD_ALLOWED_EMAILS"), name="SMD_ALLOWED_EMAILS"
            )
        )
        if any(
            len(email) > 320
            or "@" not in email
            or any(ord(character) < 33 or ord(character) == 127 for character in email)
            for email in allowed_emails
        ):
            raise SettingsError("SMD_ALLOWED_EMAILS contains an invalid email address")

        trusted_networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = ()
        identity_header = values.get(
            "SMD_TRUSTED_IDENTITY_HEADER", DEFAULT_TRUSTED_IDENTITY_HEADER
        ).strip()
        if not _HEADER_NAME_RE.fullmatch(identity_header):
            raise SettingsError(
                "SMD_TRUSTED_IDENTITY_HEADER is not a valid HTTP header name"
            )
        if identity_header.casefold() in _FORBIDDEN_IDENTITY_HEADERS:
            raise SettingsError("SMD_TRUSTED_IDENTITY_HEADER is not safe for identity")

        if auth_mode is AuthMode.CLOUDFLARE_ACCESS:
            raw_team_domain = values.get("SMD_CF_TEAM_DOMAIN")
            if raw_team_domain is None:
                raise SettingsError("SMD_CF_TEAM_DOMAIN is required")
            cf_issuer = _normalize_team_domain(raw_team_domain)
            cf_audiences = _split_values(
                values.get("SMD_CF_AUDIENCE"), name="SMD_CF_AUDIENCE"
            )
            if not cf_audiences:
                raise SettingsError("SMD_CF_AUDIENCE is required")
            if any(
                len(audience) > 512
                or any(
                    ord(character) < 33 or ord(character) == 127
                    for character in audience
                )
                for audience in cf_audiences
            ):
                raise SettingsError("SMD_CF_AUDIENCE contains an invalid audience")
        elif auth_mode is AuthMode.TRUSTED_PROXY:
            raw_cidrs = (
                _split_values(
                    values.get("SMD_TRUSTED_PROXY_CIDRS"),
                    name="SMD_TRUSTED_PROXY_CIDRS",
                )
                or DEFAULT_TRUSTED_PROXY_CIDRS
            )
            parsed_networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
            for raw_cidr in raw_cidrs:
                try:
                    parsed_networks.append(ipaddress.ip_network(raw_cidr, strict=False))
                except ValueError as exc:
                    raise SettingsError(
                        "SMD_TRUSTED_PROXY_CIDRS contains an invalid network"
                    ) from exc
            trusted_networks = tuple(dict.fromkeys(parsed_networks))

        if enabled:
            missing: list[str] = []
            if public_origin is None:
                missing.append("SMD_PUBLIC_ORIGIN")
            if auth_mode is None:
                missing.append("SMD_AUTH_MODE")
            if models_root is None:
                missing.append("ComfyUI models path or SMD_MODELS_ROOT")
            if state_directory is None:
                missing.append("ComfyUI user path or SMD_STATE_DIR")
            if missing:
                raise SettingsError(
                    "enabled downloader is missing required configuration: "
                    + ", ".join(missing)
                )

        return cls(
            enabled=enabled,
            public_origin=public_origin,
            auth_mode=auth_mode,
            models_root=models_root,
            state_directory=state_directory,
            cf_issuer=cf_issuer,
            cf_audiences=cf_audiences,
            allowed_emails=allowed_emails,
            trusted_proxy_networks=trusted_networks,
            trusted_identity_header=identity_header,
        )


@dataclass(frozen=True)
class SettingsState:
    settings: RuntimeSettings | None
    error: str | None = None

    @property
    def ready(self) -> bool:
        return bool(self.settings and self.settings.enabled and self.error is None)


def load_runtime_settings(
    env: Mapping[str, str] | None = None,
    *,
    folder_paths_module: object | None = None,
) -> SettingsState:
    """Load settings without turning extension misconfiguration into import failure."""

    try:
        return SettingsState(
            RuntimeSettings.from_env(env, folder_paths_module=folder_paths_module)
        )
    except SettingsError as exc:
        return SettingsState(None, str(exc))
    except (OSError, RuntimeError, TypeError, ValueError):
        return SettingsState(None, "runtime configuration could not be resolved")


__all__ = [
    "DEFAULT_TRUSTED_IDENTITY_HEADER",
    "DEFAULT_TRUSTED_PROXY_CIDRS",
    "AuthMode",
    "RuntimeSettings",
    "SettingsError",
    "SettingsState",
    "load_runtime_settings",
    "normalize_origin",
    "resolve_comfy_paths",
]
