from __future__ import annotations

from pathlib import Path

import pytest

from backend.security import (
    SecurityError,
    TokenSigner,
    require_subject_origin_and_csrf,
)


def test_download_token_is_bound_to_authenticated_subject(tmp_path: Path) -> None:
    signer = TokenSigner(tmp_path / "state")
    token, expires_at = signer.sign_download(
        {"filename": "model.safetensors"}, "cloudflare-access:user-1"
    )
    payload = signer.verify_download(token, "cloudflare-access:user-1")
    assert payload["filename"] == "model.safetensors"
    assert payload["exp"] == expires_at
    assert payload["_smd_purpose"] == "download"
    with pytest.raises(SecurityError, match="identity"):
        signer.verify_download(token, "cloudflare-access:user-2")


def test_bound_token_purpose_cannot_be_confused(tmp_path: Path) -> None:
    signer = TokenSigner(tmp_path / "state")
    csrf, _ = signer.issue_csrf("cloudflare-access:user-1", "https://comfy.example.com")
    with pytest.raises(SecurityError, match="identity"):
        signer.verify_download(csrf, "cloudflare-access:user-1")
    with pytest.raises(SecurityError, match="reserved"):
        signer.sign_download({"_smd_subject": "attacker"}, "cloudflare-access:user-1")


def test_csrf_is_short_lived_subject_and_exact_origin_bound(tmp_path: Path) -> None:
    signer = TokenSigner(tmp_path / "state")
    token, expires_at = signer.issue_csrf(
        "trusted-proxy:user@example.com", "https://comfy.example.com/"
    )
    payload = require_subject_origin_and_csrf(
        {
            "X-SMD-CSRF": token,
            "Origin": "https://comfy.example.com",
            "Sec-Fetch-Site": "same-origin",
        },
        signer,
        "trusted-proxy:user@example.com",
        "https://comfy.example.com",
    )
    assert payload["exp"] == expires_at
    assert expires_at > 0
    with pytest.raises(SecurityError, match="invalid CSRF"):
        require_subject_origin_and_csrf(
            {
                "X-SMD-CSRF": token,
                "Origin": "https://comfy.example.com",
            },
            signer,
            "trusted-proxy:other@example.com",
            "https://comfy.example.com",
        )
    with pytest.raises(SecurityError, match="cross-origin"):
        require_subject_origin_and_csrf(
            {"X-SMD-CSRF": token, "Origin": "https://evil.example"},
            signer,
            "trusted-proxy:user@example.com",
            "https://comfy.example.com",
        )
