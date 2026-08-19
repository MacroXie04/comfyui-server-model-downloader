from __future__ import annotations

import asyncio
import os
import socket
from pathlib import Path

import pytest

from backend import security
from backend.security import (
    SecurityError,
    SourceURL,
    TokenSigner,
    ensure_state_directory,
    require_public_dns,
    require_same_origin_and_csrf,
    resolve_model_paths,
    validate_directory,
    validate_filename,
    validate_redirect_url,
    validate_sha256,
    validate_source_url,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/"
            "split_files/vae/ae.safetensors",
            SourceURL(
                provider="huggingface",
                url=(
                    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/"
                    "split_files/vae/ae.safetensors"
                ),
                repo_id="Comfy-Org/z_image_turbo",
                revision="main",
                repo_path="split_files/vae/ae.safetensors",
            ),
        ),
        # A repository-root model is a valid HF resolve URL too.
        (
            "https://huggingface.co/org/repo/resolve/0123456789abcdef/model.safetensors",
            SourceURL(
                provider="huggingface",
                url="https://huggingface.co/org/repo/resolve/0123456789abcdef/model.safetensors",
                repo_id="org/repo",
                revision="0123456789abcdef",
                repo_path="model.safetensors",
            ),
        ),
        (
            "https://huggingface.co/org/repo/resolve/main/model.safetensors?download=true",
            SourceURL(
                provider="huggingface",
                url=(
                    "https://huggingface.co/org/repo/resolve/main/"
                    "model.safetensors?download=true"
                ),
                repo_id="org/repo",
                revision="main",
                repo_path="model.safetensors",
            ),
        ),
        (
            "https://civitai.com/api/download/models/123456",
            SourceURL(
                provider="civitai",
                url="https://civitai.com/api/download/models/123456",
                version_id=123456,
            ),
        ),
        (
            "https://civitai.com/api/download/models/123456/"
            "?type=Model&format=SafeTensor&size=full&fp=fp16",
            SourceURL(
                provider="civitai",
                url=(
                    "https://civitai.com/api/download/models/123456/"
                    "?type=Model&format=SafeTensor&size=full&fp=fp16"
                ),
                version_id=123456,
            ),
        ),
        (
            "https://civitai.com/api/download/models/123456?fileId=987654",
            SourceURL(
                provider="civitai",
                url="https://civitai.com/api/download/models/123456?fileId=987654",
                version_id=123456,
                file_id=987654,
            ),
        ),
    ],
)
def test_validate_source_url_accepts_only_canonical_sources(
    url: str, expected: SourceURL
) -> None:
    assert validate_source_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://huggingface.co/org/repo/resolve/main/model.safetensors",
        "file:///etc/passwd",
        "ftp://huggingface.co/org/repo/resolve/main/model.safetensors",
        "https://user:password@huggingface.co/org/repo/resolve/main/model.safetensors",
        "https://huggingface.co:444/org/repo/resolve/main/model.safetensors",
        "https://huggingface.co:bad/org/repo/resolve/main/model.safetensors",
        "https://huggingface.co/org/repo/resolve/main/model.safetensors#fragment",
        "https://huggingface.co.evil.example/org/repo/resolve/main/model.safetensors",
        "https://hf.co/org/repo/resolve/main/model.safetensors",
        "https://huggingface.co/org/repo/blob/main/model.safetensors",
        "https://huggingface.co/org/repo/resolve/main/model.ckpt",
        "https://huggingface.co/org/repo/resolve/main/model.sft",
        "https://huggingface.co/org/repo/resolve/main/../model.safetensors",
        "https://huggingface.co/org/repo/resolve/main/%2e%2e/model.safetensors",
        "https://huggingface.co/org/repo/resolve/main/a%2fb/model.safetensors",
        "https://huggingface.co/org/repo/resolve/main/a%5cb/model.safetensors",
        "https://huggingface.co/org/repo/resolve/main/model.safetensors?download=false",
        "https://huggingface.co/org/repo/resolve/main/model.safetensors?Download=true",
        "https://huggingface.co/org/repo/resolve/main/model.safetensors?download=true&download=true",
        "https://huggingface.co/org/repo/resolve/main/model.safetensors?download=true&x=1",
        "https://huggingface.co/org/repo/resolve/main/model.safetensors?token=secret",
        "https://civitai.com/api/v1/model-versions/123",
        "https://civitai.com/api/download/models/not-a-number",
        "https://civitai.com/api/download/models/-1",
        "https://civitai.com/api/download/models/1?token=secret",
        "https://civitai.com/api/download/models/1?API_KEY=secret",
        "https://civitai.com/api/download/models/1?redirect=https://evil.example",
        "https://civitai.com/api/download/models/1?fileId=",
        "https://civitai.com/api/download/models/1?fileId=-2",
        "https://civitai.com/api/download/models/1?fileId=abc",
        # Duplicate selector keys are ambiguous across HTTP implementations and
        # therefore must not be part of a signed canonical URL.
        "https://civitai.com/api/download/models/1?fileId=2&fileId=3",
        "https://civitai.com/api/download/models/1?type=Model&type=Checkpoint",
    ],
)
def test_validate_source_url_rejects_unsafe_or_noncanonical_urls(url: str) -> None:
    with pytest.raises(SecurityError):
        validate_source_url(url)


@pytest.mark.parametrize(
    ("url", "provider", "host"),
    [
        ("https://huggingface.co/file", "huggingface", "huggingface.co"),
        ("https://cdn-lfs.huggingface.co/file", "huggingface", "cdn-lfs.huggingface.co"),
        ("https://cas-bridge.xethub.hf.co/file", "huggingface", "cas-bridge.xethub.hf.co"),
        ("https://civitai.com/file", "civitai", "civitai.com"),
        ("https://download.civitai.com/file", "civitai", "download.civitai.com"),
    ],
)
def test_validate_redirect_url_accepts_provider_owned_hosts(
    url: str, provider: str, host: str
) -> None:
    assert validate_redirect_url(url, provider) == host


@pytest.mark.parametrize(
    ("url", "provider"),
    [
        ("https://huggingface.co.evil.example/file", "huggingface"),
        ("https://evil.example/file", "civitai"),
        ("https://127.0.0.1/file", "huggingface"),
        ("https://huggingface.co:8443/file", "huggingface"),
        ("https://huggingface.co/file?token=x", "huggingface"),
        ("https://civitai.com/file", "unknown-provider"),
    ],
)
def test_validate_redirect_url_rejects_host_confusion_and_unknown_provider(
    url: str, provider: str
) -> None:
    with pytest.raises(SecurityError):
        validate_redirect_url(url, provider)


def _dns_result(address: str) -> tuple[object, ...]:
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    sockaddr = (address, 443, 0, 0) if family == socket.AF_INET6 else (address, 443)
    return (family, socket.SOCK_STREAM, 6, "", sockaddr)


def _run_dns_check(monkeypatch: pytest.MonkeyPatch, addresses: list[str]) -> None:
    async def fake_getaddrinfo(self, host, port, **kwargs):  # noqa: ANN001, ANN202
        assert host == "cdn.example"
        assert port == 443
        return [_dns_result(address) for address in addresses]

    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", fake_getaddrinfo)
    asyncio.run(require_public_dns("cdn.example"))


def test_require_public_dns_accepts_only_all_public_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_dns_check(monkeypatch, ["8.8.8.8", "2606:4700:4700::1111"])


@pytest.mark.parametrize(
    "addresses",
    [
        ["127.0.0.1"],
        ["10.0.0.1"],
        ["169.254.169.254"],
        ["0.0.0.0"],
        ["224.0.0.1"],
        ["::1"],
        ["fe80::1"],
        ["8.8.8.8", "10.0.0.1"],
    ],
)
def test_require_public_dns_rejects_any_nonpublic_answer(
    monkeypatch: pytest.MonkeyPatch, addresses: list[str]
) -> None:
    with pytest.raises(SecurityError, match="non-public"):
        _run_dns_check(monkeypatch, addresses)


def test_require_public_dns_rejects_empty_and_resolution_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def empty(self, host, port, **kwargs):  # noqa: ANN001, ANN202
        return []

    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", empty)
    with pytest.raises(SecurityError, match="no addresses"):
        asyncio.run(require_public_dns("cdn.example"))

    async def failed(self, host, port, **kwargs):  # noqa: ANN001, ANN202
        raise socket.gaierror("no such host")

    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", failed)
    with pytest.raises(SecurityError, match="DNS resolution failed"):
        asyncio.run(require_public_dns("cdn.example"))


@pytest.mark.parametrize("directory", security.ALLOWED_DIRECTORIES)
def test_validate_directory_accepts_static_allowlist(directory: str) -> None:
    assert validate_directory(directory) == directory


@pytest.mark.parametrize(
    "directory",
    ["", "auto", "../vae", "vae/subdir", "/tmp", "custom_nodes", "outputs", None],
)
def test_validate_directory_rejects_everything_outside_allowlist(directory) -> None:  # noqa: ANN001
    with pytest.raises(SecurityError):
        validate_directory(directory)


@pytest.mark.parametrize("filename", ["model.safetensors", "MODEL.SAFETENSORS"])
def test_validate_filename_accepts_safe_basename(filename: str) -> None:
    assert validate_filename(filename) == filename


@pytest.mark.parametrize(
    "filename",
    [
        "",
        ".",
        "..",
        ".hidden.safetensors",
        "../model.safetensors",
        "subdir/model.safetensors",
        r"subdir\model.safetensors",
        "/tmp/model.safetensors",
        "model.safetensors\x00",
        "model.ckpt",
        "model.sft",
        "model.pt",
        "model.bin",
        "archive.zip",
        None,
        123,
        "x" * 256,
    ],
)
def test_validate_filename_rejects_paths_and_unsafe_formats(filename) -> None:  # noqa: ANN001
    with pytest.raises(SecurityError):
        validate_filename(filename)


def test_validate_sha256_normalizes_and_requires_full_digest() -> None:
    upper = "AB" * 32
    assert validate_sha256(upper) == upper.lower()
    for invalid in ("", "a" * 63, "a" * 65, "z" * 64, None, 123):
        with pytest.raises(SecurityError):
            validate_sha256(invalid)  # type: ignore[arg-type]


def test_resolve_model_paths_creates_only_allowlisted_child(tmp_path: Path) -> None:
    final, partial = resolve_model_paths(tmp_path / "models", "vae", "ae.safetensors")
    assert final == tmp_path / "models" / "vae" / "ae.safetensors"
    assert partial == tmp_path / "models" / "vae" / "ae.safetensors.part"
    assert final.parent.is_dir()


def test_resolve_model_paths_rejects_symlinked_root(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "models"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(SecurityError, match="models root"):
        resolve_model_paths(linked, "vae", "ae.safetensors")


def test_resolve_model_paths_rejects_symlinked_destination(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (models / "vae").symlink_to(outside, target_is_directory=True)
    with pytest.raises(SecurityError, match="destination"):
        resolve_model_paths(models, "vae", "ae.safetensors")


@pytest.mark.parametrize("partial", [False, True])
def test_resolve_model_paths_rejects_symlinked_output_files(
    tmp_path: Path, partial: bool
) -> None:
    models = tmp_path / "models"
    target_dir = models / "vae"
    target_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"do not overwrite")
    suffix = ".part" if partial else ""
    (target_dir / f"ae.safetensors{suffix}").symlink_to(outside)
    with pytest.raises(SecurityError, match="regular files"):
        resolve_model_paths(models, "vae", "ae.safetensors")
    assert outside.read_bytes() == b"do not overwrite"


def test_ensure_state_directory_rejects_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "state"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(SecurityError, match="state directory"):
        ensure_state_directory(link)


def test_token_round_trip_tamper_and_key_permissions(tmp_path: Path) -> None:
    signer = TokenSigner(tmp_path / "state")
    payload = {
        "canonical_url": "https://huggingface.co/org/repo/resolve/" + "a" * 40 + "/m.safetensors",
        "directory": "vae",
        "filename": "m.safetensors",
        "size": 4,
        "sha256": "00" * 32,
    }
    token, expires = signer.sign(payload)
    verified = signer.verify(token)
    assert {key: verified[key] for key in payload} == payload
    assert verified["exp"] == expires
    assert verified["v"] == 1
    assert os.stat(tmp_path / "state" / "download-token.key").st_mode & 0o777 == 0o600

    body, signature = token.split(".")
    replacement = "A" if body[-1] != "A" else "B"
    with pytest.raises(SecurityError, match="signature"):
        signer.verify(body[:-1] + replacement + "." + signature)
    replacement = "A" if signature[-1] != "A" else "B"
    with pytest.raises(SecurityError, match="signature"):
        signer.verify(body + "." + signature[:-1] + replacement)

    other_signer = TokenSigner(tmp_path / "other-state")
    with pytest.raises(SecurityError, match="signature"):
        other_signer.verify(token)


def test_token_expiration_boundary_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(security.time, "time", lambda: 1_000)
    signer = TokenSigner(tmp_path / "state")
    token, expires = signer.sign({"purpose": "download"}, ttl_seconds=0)
    assert expires == 1_000
    with pytest.raises(SecurityError, match="expired"):
        signer.verify(token)


@pytest.mark.parametrize(
    "token",
    ["", "abc", "a.b.c", ".", "***.abc", "abc.***", "x" * 32769],
)
def test_token_rejects_malformed_values(tmp_path: Path, token: str) -> None:
    signer = TokenSigner(tmp_path / "state")
    with pytest.raises(SecurityError):
        signer.verify(token)


def test_same_origin_and_csrf_accepts_cloudflare_forwarded_host() -> None:
    require_same_origin_and_csrf(
        {
            "X-SMD-CSRF": "csrf",
            "Origin": "https://comfy.example.com",
            "Host": "127.0.0.1:8188",
            "X-Forwarded-Host": "comfy.example.com",
            "Sec-Fetch-Site": "same-origin",
        },
        "csrf",
    )


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-SMD-CSRF": "wrong", "Origin": "https://comfy.example.com", "Host": "comfy.example.com"},
        {"X-SMD-CSRF": "csrf", "Host": "comfy.example.com"},
        {"X-SMD-CSRF": "csrf", "Origin": "http://comfy.example.com", "Host": "comfy.example.com"},
        {"X-SMD-CSRF": "csrf", "Origin": "https://evil.example", "Host": "comfy.example.com"},
        {
            "X-SMD-CSRF": "csrf",
            "Origin": "https://comfy.example.com/path",
            "Host": "comfy.example.com",
        },
        {
            "X-SMD-CSRF": "csrf",
            "Origin": "https://comfy.example.com",
            "Host": "comfy.example.com",
            "Sec-Fetch-Site": "cross-site",
        },
    ],
)
def test_same_origin_and_csrf_rejects_missing_or_cross_site_headers(headers) -> None:  # noqa: ANN001
    with pytest.raises(SecurityError):
        require_same_origin_and_csrf(headers, "csrf")
