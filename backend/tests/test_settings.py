from __future__ import annotations

import ipaddress
from pathlib import Path

import pytest

from backend.settings import (
    AuthMode,
    RuntimeSettings,
    SettingsError,
    load_runtime_settings,
    normalize_origin,
)


class _FolderPaths:
    def __init__(self, root: Path) -> None:
        self.models_dir = str(root / "models")
        self.root = root

    def get_system_user_directory(self, name: str) -> str:
        assert name == "server_model_downloader"
        return str(self.root / "user" / name)


def _base_env(root: Path) -> dict[str, str]:
    return {
        "SMD_ENABLED": "true",
        "SMD_PUBLIC_ORIGIN": "https://comfy.example.com",
        "SMD_MODELS_ROOT": str(root / "models"),
        "SMD_STATE_DIR": str(root / "state"),
    }


def test_disabled_by_default_and_import_safe_without_comfy_paths() -> None:
    settings = RuntimeSettings.from_env({}, folder_paths_module=object())
    assert settings.enabled is False
    assert settings.models_root is None
    assert settings.state_directory is None
    assert settings.auth_mode is None


def test_paths_use_env_overrides_before_comfy_native_paths(tmp_path: Path) -> None:
    env = _base_env(tmp_path / "override")
    env.update(
        {
            "SMD_AUTH_MODE": "trusted-proxy",
            "SMD_TRUSTED_PROXY_CIDRS": "10.0.0.4/24,2001:db8::1/64",
            "SMD_TRUSTED_IDENTITY_HEADER": "X-Authenticated-User",
        }
    )
    settings = RuntimeSettings.from_env(
        env, folder_paths_module=_FolderPaths(tmp_path / "comfy")
    )
    assert settings.models_root == tmp_path / "override" / "models"
    assert settings.state_directory == tmp_path / "override" / "state"
    assert settings.auth_mode is AuthMode.TRUSTED_PROXY
    assert settings.trusted_proxy_networks == (
        ipaddress.ip_network("10.0.0.0/24"),
        ipaddress.ip_network("2001:db8::/64"),
    )
    assert settings.trusted_identity_header == "X-Authenticated-User"


def test_paths_fall_back_to_comfy_native_locations(tmp_path: Path) -> None:
    env = {
        "SMD_ENABLED": "1",
        "SMD_PUBLIC_ORIGIN": "https://comfy.example.com/",
        "SMD_AUTH_MODE": "trusted-proxy",
    }
    settings = RuntimeSettings.from_env(env, folder_paths_module=_FolderPaths(tmp_path))
    assert settings.public_origin == "https://comfy.example.com"
    assert settings.models_root == tmp_path / "models"
    assert settings.state_directory == tmp_path / "user" / "server_model_downloader"
    assert settings.trusted_proxy_networks == (
        ipaddress.ip_network("127.0.0.0/8"),
        ipaddress.ip_network("::1/128"),
    )


def test_cloudflare_settings_are_normalized_and_typed(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    env.update(
        {
            "SMD_AUTH_MODE": "cloudflare-access",
            "SMD_CF_TEAM_DOMAIN": "https://My-Team.cloudflareaccess.com/",
            "SMD_CF_AUDIENCE": "aud-one, aud-two",
            "SMD_ALLOWED_EMAILS": "OWNER@EXAMPLE.COM,viewer@example.com",
        }
    )
    settings = RuntimeSettings.from_env(env)
    assert settings.cf_issuer == "https://my-team.cloudflareaccess.com"
    assert settings.cf_jwks_url == (
        "https://my-team.cloudflareaccess.com/cdn-cgi/access/certs"
    )
    assert settings.cf_audiences == ("aud-one", "aud-two")
    assert settings.allowed_emails == {
        "owner@example.com",
        "viewer@example.com",
    }


@pytest.mark.parametrize(
    "origin",
    [
        "http://comfy.example.com",
        "https://user@example.com",
        "https://comfy.example.com/path",
        "https://comfy.example.com?x=1",
        "https://comfy.example.com#fragment",
        "https://comfy.example.com:8443",
        "https://comfy.example.com.",
        " https://comfy.example.com",
    ],
)
def test_public_origin_is_exact_root_https(origin: str) -> None:
    with pytest.raises(SettingsError):
        normalize_origin(origin)


@pytest.mark.parametrize(
    "updates",
    [
        {},
        {"SMD_AUTH_MODE": "unknown"},
        {"SMD_AUTH_MODE": "cloudflare-access"},
        {
            "SMD_AUTH_MODE": "cloudflare-access",
            "SMD_CF_TEAM_DOMAIN": "team.example.com",
            "SMD_CF_AUDIENCE": "aud",
        },
        {
            "SMD_AUTH_MODE": "cloudflare-access",
            "SMD_CF_TEAM_DOMAIN": "team.cloudflareaccess.com",
        },
        {
            "SMD_AUTH_MODE": "trusted-proxy",
            "SMD_TRUSTED_PROXY_CIDRS": "not-a-network",
        },
        {
            "SMD_AUTH_MODE": "trusted-proxy",
            "SMD_TRUSTED_IDENTITY_HEADER": "Authorization",
        },
    ],
)
def test_enabled_configuration_fails_closed(
    tmp_path: Path, updates: dict[str, str]
) -> None:
    env = _base_env(tmp_path)
    env.pop("SMD_PUBLIC_ORIGIN")
    env.update(updates)
    with pytest.raises(SettingsError):
        RuntimeSettings.from_env(env)


def test_safe_loader_captures_configuration_error_for_degraded_api(
    tmp_path: Path,
) -> None:
    state = load_runtime_settings(
        {
            "SMD_ENABLED": "true",
            "SMD_MODELS_ROOT": str(tmp_path / "models"),
            "SMD_STATE_DIR": str(tmp_path / "state"),
        }
    )
    assert state.settings is None
    assert state.ready is False
    assert state.error and "SMD_PUBLIC_ORIGIN" in state.error


def test_safe_loader_never_leaks_origin_or_folder_path_conversion_errors() -> None:
    malformed_origin = load_runtime_settings(
        {"SMD_ENABLED": "false", "SMD_PUBLIC_ORIGIN": "https://[::1"},
        folder_paths_module=object(),
    )
    assert malformed_origin.ready is False
    assert malformed_origin.error

    class InvalidFolderPaths:
        models_dir = object()

        @staticmethod
        def get_system_user_directory(name: str) -> object:
            return object()

    invalid_paths = load_runtime_settings({}, folder_paths_module=InvalidFolderPaths())
    assert invalid_paths.ready is False
    assert invalid_paths.error
