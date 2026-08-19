from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import quote, urljoin, urlsplit

from .security import (
    ALLOWED_DIRECTORIES,
    SecurityError,
    SourceURL,
    TokenSigner,
    require_public_dns,
    validate_directory,
    validate_filename,
    validate_redirect_url,
    validate_sha256,
    validate_source_url,
)


MAX_METADATA_BYTES = 10 * 1024 * 1024
MAX_REDIRECTS = 8


class MetadataError(ValueError):
    pass


@dataclass(frozen=True)
class Candidate:
    provider: str
    requested_name: str
    filename: str
    source_filename: str
    directory: str
    canonical_url: str
    revision: str
    size: int
    sha256: str
    license: Any
    license_url: str | None
    metadata: Mapping[str, Any]

    @property
    def relative_path(self) -> str:
        return f"{self.directory}/{self.filename}"

    def token_payload(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "canonical_url": self.canonical_url,
            "revision": self.revision,
            "size": self.size,
            "sha256": self.sha256,
            "directory": self.directory,
            "filename": self.filename,
            "path": self.relative_path,
            "license": self.license,
            "license_url": self.license_url,
        }

    def public_dict(self, signer: TokenSigner) -> dict[str, Any]:
        token, expires_at = signer.sign(self.token_payload())
        return {
            "provider": self.provider,
            "requested_name": self.requested_name,
            "filename": self.filename,
            "source_filename": self.source_filename,
            "directory": self.directory,
            "relative_path": self.relative_path,
            "canonical_url": self.canonical_url,
            "revision": self.revision,
            "size": self.size,
            "sha256": self.sha256,
            "license": self.license,
            "license_url": self.license_url,
            "metadata": dict(self.metadata),
            "expires_at": expires_at,
            "download_token": token,
        }


def infer_directory(filename: str, provider_metadata: Mapping[str, Any]) -> str:
    explicit_path = str(provider_metadata.get("repo_path") or "")
    first = explicit_path.split("/", 1)[0]
    if first in ALLOWED_DIRECTORIES:
        return first
    model_type = str(provider_metadata.get("model_type") or "").lower()
    type_map = {
        "checkpoint": "checkpoints",
        "lora": "loras",
        "locon": "loras",
        "lycoris": "loras",
        "controlnet": "controlnet",
        "textualinversion": "embeddings",
        "vae": "vae",
        "upscaler": "upscale_models",
    }
    if model_type in type_map:
        return type_map[model_type]
    value = f"{filename} {explicit_path}".lower()
    heuristics = (
        (("clip_vision", "vision_encoder"), "clip_vision"),
        (("audio_encoder", "wav2vec"), "audio_encoders"),
        (("text_encoder", "qwen", "t5xxl", "umt5", "clip_l", "clip_g"), "text_encoders"),
        (("vae", "ae.safetensors", "autoencoder"), "vae"),
        (("lora", "lycoris", "locon"), "loras"),
        (("controlnet", "control_net"), "controlnet"),
        (("upscale", "ultrasharp", "esrgan", "swinir"), "upscale_models"),
        (("checkpoint",), "checkpoints"),
    )
    for needles, directory in heuristics:
        if any(needle in value for needle in needles):
            return directory
    return "diffusion_models"


def _provider_headers(provider: str, include_auth: bool = False) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "identity",
        "User-Agent": "ComfyUI-ServerModelDownloader/1.0",
    }
    if include_auth and provider == "huggingface":
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    elif include_auth and provider == "civitai":
        token = os.environ.get("CIVITAI_API_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return headers


async def _read_limited(response: Any, limit: int = MAX_METADATA_BYTES) -> bytes:
    length = response.headers.get("Content-Length")
    if length:
        try:
            parsed_length = int(length)
        except ValueError as exc:
            raise MetadataError("provider returned an invalid Content-Length") from exc
        if parsed_length > limit:
            raise MetadataError("metadata response is too large")
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.content.iter_chunked(64 * 1024):
        total += len(chunk)
        if total > limit:
            raise MetadataError("metadata response is too large")
        chunks.append(chunk)
    return b"".join(chunks)


async def safe_fetch_json(session: Any, url: str, provider: str) -> Any:
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        host = validate_redirect_url(current, provider)
        await require_public_dns(host)
        origin_host = "huggingface.co" if provider == "huggingface" else "civitai.com"
        headers = _provider_headers(provider, include_auth=host == origin_host)
        response = await session.get(current, headers=headers, allow_redirects=False)
        try:
            if response.status in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                if not location:
                    raise MetadataError("redirect is missing Location")
                current = urljoin(current, location)
                validate_redirect_url(current, provider)
                continue
            if response.status != 200:
                body = (await _read_limited(response, 64 * 1024)).decode("utf-8", "replace")
                raise MetadataError(f"metadata request failed with HTTP {response.status}: {body[:300]}")
            raw = await _read_limited(response)
            try:
                return json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise MetadataError("provider returned invalid JSON metadata") from exc
        finally:
            response.release()
    raise MetadataError("too many metadata redirects")


class MetadataInspector:
    def __init__(self, signer: TokenSigner):
        self.signer = signer

    async def inspect(self, session: Any, model: Mapping[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(model, Mapping):
            raise MetadataError("each model must be an object")
        requested_name = validate_filename(model.get("name"))
        source = validate_source_url(model.get("url"))
        requested_directory = model.get("directory", "auto")
        if requested_directory != "auto":
            validate_directory(requested_directory)
        if source.provider == "huggingface":
            candidates = await self._inspect_huggingface(
                session, source, requested_name, requested_directory
            )
        else:
            candidates = await self._inspect_civitai(
                session, source, requested_name, requested_directory
            )
        return [candidate.public_dict(self.signer) for candidate in candidates]

    async def _inspect_huggingface(
        self,
        session: Any,
        source: SourceURL,
        requested_name: str,
        requested_directory: str,
    ) -> list[Candidate]:
        assert source.repo_id and source.revision and source.repo_path
        api_url = (
            "https://huggingface.co/api/models/"
            f"{quote(source.repo_id, safe='/')}/revision/{quote(source.revision, safe='')}?blobs=true"
        )
        document = await safe_fetch_json(session, api_url, "huggingface")
        if not isinstance(document, dict):
            raise MetadataError("unexpected Hugging Face metadata")
        resolved_revision = document.get("sha")
        if not isinstance(resolved_revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", resolved_revision):
            raise MetadataError("Hugging Face did not return an immutable revision")
        sibling = None
        for item in document.get("siblings") or []:
            if isinstance(item, dict) and item.get("rfilename") == source.repo_path:
                sibling = item
                break
        if sibling is None:
            raise MetadataError("file is not present in the Hugging Face repository revision")
        source_filename = validate_filename(source.repo_path.rsplit("/", 1)[-1])
        lfs = sibling.get("lfs") if isinstance(sibling.get("lfs"), dict) else {}
        sha256 = validate_sha256(lfs.get("sha256"))
        size = lfs.get("size") or sibling.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise MetadataError("Hugging Face did not provide a valid file size")
        card_data = document.get("cardData") if isinstance(document.get("cardData"), dict) else {}
        license_value: Any = card_data.get("license") or "unknown"
        license_url = card_data.get("license_link")
        if (
            not isinstance(license_url, str)
            or urlsplit(license_url).scheme != "https"
            or not urlsplit(license_url).hostname
        ):
            license_url = f"https://huggingface.co/{source.repo_id}/blob/{resolved_revision}/LICENSE"
        provider_metadata = {
            "repo_id": source.repo_id,
            "repo_path": source.repo_path,
            "requested_revision": source.revision,
            "resolved_revision": resolved_revision,
            "model_id": document.get("id"),
            "last_modified": document.get("lastModified"),
        }
        directory = (
            infer_directory(requested_name, provider_metadata)
            if requested_directory == "auto"
            else requested_directory
        )
        canonical_url = (
            f"https://huggingface.co/{quote(source.repo_id, safe='/')}/resolve/"
            f"{quote(resolved_revision, safe='')}/{quote(source.repo_path, safe='/')}"
        )
        # Revalidate the URL that will actually be signed and downloaded.
        validate_source_url(canonical_url)
        return [
            Candidate(
                provider="huggingface",
                requested_name=requested_name,
                filename=requested_name,
                source_filename=source_filename,
                directory=directory,
                canonical_url=canonical_url,
                revision=resolved_revision,
                size=size,
                sha256=sha256,
                license=license_value,
                license_url=license_url,
                metadata=provider_metadata,
            )
        ]

    async def _inspect_civitai(
        self,
        session: Any,
        source: SourceURL,
        requested_name: str,
        requested_directory: str,
    ) -> list[Candidate]:
        assert source.version_id is not None
        version_url = f"https://civitai.com/api/v1/model-versions/{source.version_id}"
        version = await safe_fetch_json(session, version_url, "civitai")
        if not isinstance(version, dict) or version.get("id") != source.version_id:
            raise MetadataError("Civitai model-version metadata did not match the download URL")
        files = []
        for item in version.get("files") or []:
            if not isinstance(item, dict):
                continue
            try:
                source_filename = validate_filename(item.get("name"))
                sha256 = validate_sha256((item.get("hashes") or {}).get("SHA256"))
            except (SecurityError, AttributeError):
                continue
            if source.file_id is not None and item.get("id") != source.file_id:
                continue
            if source_filename.casefold() == requested_name.casefold():
                files.append((item, source_filename, sha256))
        if not files:
            safe_files = []
            for item in version.get("files") or []:
                if not isinstance(item, dict):
                    continue
                try:
                    if source.file_id is not None and item.get("id") != source.file_id:
                        continue
                    safe_files.append(
                        (
                            item,
                            validate_filename(item.get("name")),
                            validate_sha256((item.get("hashes") or {}).get("SHA256")),
                        )
                    )
                except (SecurityError, AttributeError):
                    continue
            if len(safe_files) == 1:
                files = safe_files
            else:
                raise MetadataError("requested filename does not uniquely match a Civitai safetensors file")
        model_id = version.get("modelId")
        model_document: dict[str, Any] = {}
        if isinstance(model_id, int) and not isinstance(model_id, bool):
            result = await safe_fetch_json(
                session, f"https://civitai.com/api/v1/models/{model_id}", "civitai"
            )
            if isinstance(result, dict):
                model_document = result
        license_details = {
            "allow_no_credit": model_document.get("allowNoCredit"),
            "allow_commercial_use": model_document.get("allowCommercialUse"),
            "allow_derivatives": model_document.get("allowDerivatives"),
            "allow_different_license": model_document.get("allowDifferentLicense"),
        }
        license_value = (
            "Civitai terms: "
            f"commercial={license_details['allow_commercial_use']}, "
            f"credit-required={not license_details['allow_no_credit'] if isinstance(license_details['allow_no_credit'], bool) else 'unknown'}, "
            f"derivatives={license_details['allow_derivatives']}, "
            f"different-license={license_details['allow_different_license']}"
        )
        license_url = f"https://civitai.com/models/{model_id}" if isinstance(model_id, int) else None
        if len(files) != 1:
            raise MetadataError("requested filename matches more than one Civitai file")
        candidates: list[Candidate] = []
        for item, source_filename, sha256 in files:
            download_url = item.get("downloadUrl")
            if not isinstance(download_url, str):
                raise MetadataError("Civitai did not provide a download URL")
            parsed_download = validate_source_url(download_url)
            if parsed_download.provider != "civitai" or parsed_download.version_id != source.version_id:
                raise MetadataError("Civitai file URL does not match the inspected model version")
            item_id = item.get("id")
            if not isinstance(item_id, int) or isinstance(item_id, bool):
                raise MetadataError("Civitai file metadata is missing a numeric file id")
            if parsed_download.file_id is not None and parsed_download.file_id != item_id:
                raise MetadataError("Civitai download URL fileId does not match API metadata")
            if source.file_id is not None and source.file_id != item_id:
                raise MetadataError("requested Civitai fileId does not match API metadata")
            size_kb = item.get("sizeKB")
            if not isinstance(size_kb, (int, float)) or isinstance(size_kb, bool) or size_kb <= 0:
                raise MetadataError("Civitai did not provide a valid file size")
            size = int(round(float(size_kb) * 1024))
            provider_metadata = {
                "version_id": source.version_id,
                "model_id": model_id,
                "model_name": (version.get("model") or {}).get("name")
                if isinstance(version.get("model"), dict)
                else model_document.get("name"),
                "model_type": (version.get("model") or {}).get("type")
                if isinstance(version.get("model"), dict)
                else model_document.get("type"),
                "base_model": version.get("baseModel"),
                "published_at": version.get("publishedAt"),
                "size_source": "Civitai API sizeKB",
                "license_details": license_details,
            }
            directory = (
                infer_directory(requested_name, provider_metadata)
                if requested_directory == "auto"
                else requested_directory
            )
            # The Civitai API commonly omits ``fileId`` from ``downloadUrl``
            # even when a model version has multiple files.  Bind the signed
            # job to the exact API-verified file instead of relying on mutable
            # type/format selector heuristics.
            canonical_url = (
                f"https://civitai.com/api/download/models/{source.version_id}"
                f"?fileId={item_id}"
            )
            canonical_source = validate_source_url(canonical_url)
            if (
                canonical_source.provider != "civitai"
                or canonical_source.version_id != source.version_id
                or canonical_source.file_id != item_id
            ):
                raise MetadataError("failed to bind the Civitai file download URL")
            candidates.append(
                Candidate(
                    provider="civitai",
                    requested_name=requested_name,
                    filename=requested_name,
                    source_filename=source_filename,
                    directory=directory,
                    canonical_url=canonical_url,
                    revision=str(source.version_id),
                    size=size,
                    sha256=sha256,
                    license=license_value,
                    license_url=license_url,
                    metadata=provider_metadata,
                )
            )
        return candidates
