from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from backend import metadata
from backend.metadata import (
    MAX_METADATA_BYTES,
    MetadataError,
    MetadataInspector,
    _read_limited,
    infer_directory,
    safe_fetch_json,
)
from backend.security import SecurityError, TokenSigner


def _run(coro):  # noqa: ANN001, ANN202
    return asyncio.run(coro)


@pytest.mark.parametrize(
    ("filename", "provider_metadata", "expected"),
    [
        ("anything.safetensors", {"repo_path": "vae/anything.safetensors"}, "vae"),
        ("anything.safetensors", {"repo_path": "text_encoders/anything.safetensors"}, "text_encoders"),
        ("anything.safetensors", {"model_type": "Checkpoint"}, "checkpoints"),
        ("anything.safetensors", {"model_type": "LORA"}, "loras"),
        ("anything.safetensors", {"model_type": "ControlNet"}, "controlnet"),
        ("anything.safetensors", {"model_type": "TextualInversion"}, "embeddings"),
        ("anything.safetensors", {"model_type": "VAE"}, "vae"),
        ("4x-upscale.safetensors", {}, "upscale_models"),
        ("qwen_3_4b.safetensors", {}, "text_encoders"),
        ("clip_vision_h.safetensors", {}, "clip_vision"),
        ("vae.safetensors", {}, "vae"),
        ("foo-controlnet.safetensors", {}, "controlnet"),
        ("model.safetensors", {}, "diffusion_models"),
    ],
)
def test_infer_directory_uses_explicit_repo_path_then_metadata_then_name(
    filename: str, provider_metadata: dict[str, Any], expected: str
) -> None:
    assert infer_directory(filename, provider_metadata) == expected


def _hf_document(
    *,
    repo_path: str = "split_files/vae/ae.safetensors",
    sha: str = "a" * 40,
    digest: str = "B" * 64,
    size: object = 1024,
    license_value: object = "apache-2.0",
    license_link: object = None,
) -> dict[str, Any]:
    lfs: dict[str, Any] = {"sha256": digest, "size": size}
    return {
        "id": "Comfy-Org/z_image_turbo",
        "sha": sha,
        "lastModified": "2026-01-01T00:00:00Z",
        "cardData": {"license": license_value, "license_link": license_link},
        "siblings": [{"rfilename": repo_path, "lfs": lfs}],
    }


def test_huggingface_inspection_pins_revision_hash_size_directory_and_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _hf_document()
    calls: list[tuple[str, str]] = []

    async def fake_fetch(session, url: str, provider: str):  # noqa: ANN001, ANN202
        calls.append((url, provider))
        return document

    monkeypatch.setattr(metadata, "safe_fetch_json", fake_fetch)
    signer = TokenSigner(tmp_path / "state")
    inspector = MetadataInspector(signer)
    [candidate] = _run(
        inspector.inspect(
            object(),
            {
                "name": "ae.safetensors",
                "url": (
                    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/"
                    "split_files/vae/ae.safetensors"
                ),
                "directory": "auto",
            },
        )
    )

    assert calls == [
        (
            "https://huggingface.co/api/models/Comfy-Org/z_image_turbo/revision/main?blobs=true",
            "huggingface",
        )
    ]
    assert candidate["provider"] == "huggingface"
    assert candidate["filename"] == "ae.safetensors"
    assert candidate["source_filename"] == "ae.safetensors"
    assert candidate["directory"] == "vae"
    assert candidate["relative_path"] == "vae/ae.safetensors"
    assert candidate["revision"] == "a" * 40
    assert candidate["canonical_url"] == (
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/"
        + "a" * 40
        + "/split_files/vae/ae.safetensors"
    )
    assert candidate["size"] == 1024
    assert candidate["sha256"] == "b" * 64
    assert candidate["license"] == "apache-2.0"
    assert candidate["license_url"] == (
        "https://huggingface.co/Comfy-Org/z_image_turbo/blob/" + "a" * 40 + "/LICENSE"
    )
    token = signer.verify(candidate["download_token"])
    assert token["canonical_url"] == candidate["canonical_url"]
    assert token["revision"] == candidate["revision"]
    assert token["size"] == candidate["size"]
    assert token["sha256"] == candidate["sha256"]
    assert token["directory"] == "vae"
    assert token["filename"] == "ae.safetensors"
    assert token["path"] == "vae/ae.safetensors"


def test_huggingface_explicit_directory_overrides_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_fetch(session, url, provider):  # noqa: ANN001, ANN202
        return _hf_document(repo_path="model.safetensors")

    monkeypatch.setattr(metadata, "safe_fetch_json", fake_fetch)
    inspector = MetadataInspector(TokenSigner(tmp_path / "state"))
    [candidate] = _run(
        inspector.inspect(
            object(),
            {
                "name": "renamed.safetensors",
                "url": "https://huggingface.co/org/repo/resolve/main/model.safetensors",
                "directory": "checkpoints",
            },
        )
    )
    assert candidate["directory"] == "checkpoints"
    assert candidate["filename"] == "renamed.safetensors"
    assert candidate["source_filename"] == "model.safetensors"


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ([], "unexpected"),
        (_hf_document(sha="main"), "immutable revision"),
        ({**_hf_document(), "siblings": []}, "not present"),
        (_hf_document(digest="bad"), "SHA256"),
        (_hf_document(size=0), "file size"),
        (_hf_document(size=True), "file size"),
        (_hf_document(size="1024"), "file size"),
    ],
)
def test_huggingface_inspection_rejects_unverifiable_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document: object,
    message: str,
) -> None:
    async def fake_fetch(session, url, provider):  # noqa: ANN001, ANN202
        return document

    monkeypatch.setattr(metadata, "safe_fetch_json", fake_fetch)
    inspector = MetadataInspector(TokenSigner(tmp_path / "state"))
    with pytest.raises((MetadataError, SecurityError), match=message):
        _run(
            inspector.inspect(
                object(),
                {
                    "name": "ae.safetensors",
                    "url": (
                        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/"
                        "split_files/vae/ae.safetensors"
                    ),
                },
            )
        )


def _civitai_version(
    *,
    version_id: object = 123,
    file_id: object = 456,
    filename: object = "model.safetensors",
    download_url: object = "https://civitai.com/api/download/models/123",
    digest: object = "C" * 64,
    size_kb: object = 1.5,
) -> dict[str, Any]:
    return {
        "id": version_id,
        "modelId": 42,
        "baseModel": "SDXL 1.0",
        "publishedAt": "2026-01-01T00:00:00Z",
        "model": {"name": "Example", "type": "LORA"},
        "files": [
            {
                "id": file_id,
                "name": filename,
                "downloadUrl": download_url,
                "hashes": {"SHA256": digest},
                "sizeKB": size_kb,
            }
        ],
    }


def _civitai_model() -> dict[str, Any]:
    return {
        "id": 42,
        "name": "Example",
        "type": "LORA",
        "allowNoCredit": True,
        "allowCommercialUse": ["Image", "RentCivit"],
        "allowDerivatives": False,
        "allowDifferentLicense": False,
    }


def test_civitai_file_id_is_bound_to_metadata_url_hash_size_and_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    version = _civitai_version()
    calls: list[tuple[str, str]] = []

    async def fake_fetch(session, url: str, provider: str):  # noqa: ANN001, ANN202
        calls.append((url, provider))
        if url.endswith("/model-versions/123"):
            return version
        if url.endswith("/models/42"):
            return _civitai_model()
        raise AssertionError(url)

    monkeypatch.setattr(metadata, "safe_fetch_json", fake_fetch)
    signer = TokenSigner(tmp_path / "state")
    inspector = MetadataInspector(signer)
    [candidate] = _run(
        inspector.inspect(
            object(),
            {
                "name": "model.safetensors",
                "url": "https://civitai.com/api/download/models/123",
                "directory": "auto",
            },
        )
    )
    assert calls == [
        ("https://civitai.com/api/v1/model-versions/123", "civitai"),
        ("https://civitai.com/api/v1/models/42", "civitai"),
    ]
    assert candidate["canonical_url"] == (
        "https://civitai.com/api/download/models/123?fileId=456"
    )
    assert candidate["revision"] == "123"
    assert candidate["directory"] == "loras"
    assert candidate["size"] == 1536
    assert candidate["sha256"] == "c" * 64
    assert "commercial=['Image', 'RentCivit']" in candidate["license"]
    assert "credit-required=False" in candidate["license"]
    assert candidate["license_url"] == "https://civitai.com/models/42"
    token = signer.verify(candidate["download_token"])
    assert token["canonical_url"] == candidate["canonical_url"]
    assert token["sha256"] == "c" * 64


@pytest.mark.parametrize(
    ("source_url", "version", "message"),
    [
        (
            "https://civitai.com/api/download/models/999?fileId=456",
            _civitai_version(),
            "did not match",
        ),
        (
            "https://civitai.com/api/download/models/123?fileId=999",
            _civitai_version(),
            "filename does not uniquely match|fileId",
        ),
        (
            "https://civitai.com/api/download/models/123?fileId=456",
            _civitai_version(
                download_url="https://civitai.com/api/download/models/123?fileId=999"
            ),
            "fileId",
        ),
        (
            "https://civitai.com/api/download/models/123?fileId=456",
            _civitai_version(download_url="https://civitai.com/api/download/models/124?fileId=456"),
            "model version",
        ),
        (
            "https://civitai.com/api/download/models/123?fileId=456",
            _civitai_version(file_id="456"),
            "numeric file id|uniquely match",
        ),
        (
            "https://civitai.com/api/download/models/123?fileId=456",
            _civitai_version(digest="bad"),
            "uniquely match|SHA256",
        ),
        (
            "https://civitai.com/api/download/models/123?fileId=456",
            _civitai_version(size_kb=0),
            "file size",
        ),
        (
            "https://civitai.com/api/download/models/123?fileId=456",
            _civitai_version(size_kb=True),
            "file size",
        ),
    ],
)
def test_civitai_inspection_rejects_unbound_or_unverifiable_file_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_url: str,
    version: dict[str, Any],
    message: str,
) -> None:
    async def fake_fetch(session, url: str, provider: str):  # noqa: ANN001, ANN202
        if "/model-versions/" in url:
            return version
        return _civitai_model()

    monkeypatch.setattr(metadata, "safe_fetch_json", fake_fetch)
    inspector = MetadataInspector(TokenSigner(tmp_path / "state"))
    with pytest.raises((MetadataError, SecurityError), match=message):
        _run(
            inspector.inspect(
                object(),
                {"name": "model.safetensors", "url": source_url, "directory": "auto"},
            )
        )


class _Content:
    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks

    async def iter_chunked(self, size: int):  # noqa: ANN201
        for chunk in self.chunks:
            yield chunk


class _Response:
    def __init__(self, status: int, *, headers: dict[str, str] | None = None, chunks=None):  # noqa: ANN001
        self.status = status
        self.headers = headers or {}
        self.content = _Content(chunks or [])
        self.released = False

    def release(self) -> None:
        self.released = True


class _Session:
    def __init__(self, responses: list[_Response]):
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def get(self, url: str, **kwargs):  # noqa: ANN201
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def test_read_limited_checks_declared_and_streamed_size() -> None:
    response = _Response(200, headers={"Content-Length": str(MAX_METADATA_BYTES + 1)})
    with pytest.raises(MetadataError, match="too large"):
        _run(_read_limited(response))

    response = _Response(200, headers={"Content-Length": "not-a-number"})
    with pytest.raises(MetadataError, match="Content-Length"):
        _run(_read_limited(response))

    response = _Response(200, chunks=[b"a" * 6, b"b" * 5])
    with pytest.raises(MetadataError, match="too large"):
        _run(_read_limited(response, 10))


def test_safe_fetch_json_revalidates_dns_and_redirect_host_each_hop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _Response(302, headers={"Location": "https://cdn-lfs.huggingface.co/meta"})
    second = _Response(200, chunks=[json.dumps({"ok": True}).encode()])
    session = _Session([first, second])
    dns_hosts: list[str] = []

    async def fake_dns(host: str, port: int = 443) -> None:
        dns_hosts.append(host)

    monkeypatch.setattr(metadata, "require_public_dns", fake_dns)
    result = _run(
        safe_fetch_json(session, "https://huggingface.co/api/models/org/repo", "huggingface")
    )
    assert result == {"ok": True}
    assert dns_hosts == ["huggingface.co", "cdn-lfs.huggingface.co"]
    assert [call[0] for call in session.calls] == [
        "https://huggingface.co/api/models/org/repo",
        "https://cdn-lfs.huggingface.co/meta",
    ]
    assert all(call[1]["allow_redirects"] is False for call in session.calls)
    assert first.released and second.released


def test_safe_fetch_json_rejects_bad_redirect_status_and_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_dns(host: str, port: int = 443) -> None:
        return None

    monkeypatch.setattr(metadata, "require_public_dns", fake_dns)

    missing_location = _Response(302)
    with pytest.raises(MetadataError, match="missing Location"):
        _run(
            safe_fetch_json(
                _Session([missing_location]),
                "https://huggingface.co/api/models/org/repo",
                "huggingface",
            )
        )
    assert missing_location.released

    error = _Response(403, chunks=[b"denied"])
    with pytest.raises(MetadataError, match="HTTP 403"):
        _run(
            safe_fetch_json(
                _Session([error]),
                "https://huggingface.co/api/models/org/repo",
                "huggingface",
            )
        )
    assert error.released

    invalid = _Response(200, chunks=[b"not-json"])
    with pytest.raises(MetadataError, match="invalid JSON"):
        _run(
            safe_fetch_json(
                _Session([invalid]),
                "https://huggingface.co/api/models/org/repo",
                "huggingface",
            )
        )
    assert invalid.released


def test_safe_fetch_json_rejects_redirect_to_unapproved_host_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _Response(302, headers={"Location": "https://evil.example/meta"})
    session = _Session([first])

    async def fake_dns(host: str, port: int = 443) -> None:
        return None

    monkeypatch.setattr(metadata, "require_public_dns", fake_dns)
    with pytest.raises(SecurityError, match="redirect host"):
        _run(
            safe_fetch_json(
                session,
                "https://huggingface.co/api/models/org/repo",
                "huggingface",
            )
        )
    assert len(session.calls) == 1
    assert first.released
