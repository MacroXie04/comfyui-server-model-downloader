# Server Model Downloader frontend

This directory contains the browser-side half of the restricted AWS model downloader for the installed ComfyUI stack:

- ComfyUI `v0.33.1`
- ComfyUI frontend `v1.48.7`
- ComfyUI-Manager `v4.2.2`

It does not contain or modify the downloader backend.

## Files

- `server-model-downloader.js` registers the **下载到 AWS** sidebar tab and Extensions menu command.
- `server-model-downloader.css` styles the sidebar without external dependencies.
- `model-scan.mjs` extracts only model metadata matching active node selections.

## Frontend API choices

The implementation uses APIs present in the official `v1.48.7` frontend:

- Standard extension loading through `app.registerExtension()` and `../../scripts/app.js`.
- Custom sidebar registration through `app.extensionManager.registerSidebarTab()`.
- Top-menu registration through extension `commands` and `menuCommands`.
- The live graph is read from `app.rootGraph`, with `app.graph` retained as a compatibility fallback.
- Ordinary nested graphs are traversed using the same `node.subgraph` shape used by the official graph traversal utilities.
- Model metadata follows the official workflow `ModelFile` fields: `name`, `url`, `directory`, with optional metadata deliberately left for the server to resolve.
- After a completed job, `app.refreshComboInNodes()` reloads node definitions and refreshes model combo widgets throughout the root graph and subgraphs.
- HTTP requests use ComfyUI's same-origin `api.fetchApi()` helper.

Relevant official references:

- [JavaScript extensions](https://docs.comfy.org/custom-nodes/js/javascript_overview)
- [Sidebar Tabs API](https://docs.comfy.org/custom-nodes/js/javascript_sidebar_tabs)
- [Topbar Menu API](https://docs.comfy.org/custom-nodes/js/javascript_topbar_menu)
- [Frontend v1.48.7 extension types](https://github.com/Comfy-Org/ComfyUI_frontend/blob/v1.48.7/src/types/extensionTypes.ts)
- [Frontend v1.48.7 graph traversal](https://github.com/Comfy-Org/ComfyUI_frontend/blob/v1.48.7/src/utils/graphTraversalUtil.ts)
- [Frontend v1.48.7 workflow model schema](https://github.com/Comfy-Org/ComfyUI_frontend/blob/v1.48.7/src/platform/workflow/validation/schemas/workflowSchema.ts)

## Required backend contract

All endpoints are same-origin and rooted at `/server-model-downloader`.

### Session

`GET /server-model-downloader/session`

```json
{
  "csrf_token": "opaque-token",
  "allowed_directories": ["checkpoints", "diffusion_models", "text_encoders", "vae"],
  "safe_extensions": [".safetensors"]
}
```

### Inspect current workflow models

`POST /server-model-downloader/inspect` with `X-SMD-CSRF` and JSON:

```json
{
  "models": [
    {
      "name": "model.safetensors",
      "url": "https://huggingface.co/example/repo/resolve/main/model.safetensors",
      "directory": "diffusion_models"
    }
  ]
}
```

Expected response:

```json
{
  "models": [
    {
      "id": "opaque-model-id",
      "name": "model.safetensors",
      "filename": "model.safetensors",
      "url": "https://huggingface.co/example/repo/resolve/main/model.safetensors",
      "directory": "diffusion_models",
      "installed": false,
      "eligible": true,
      "reason": "",
      "source": "Hugging Face",
      "size": 123456789,
      "sha256": "optional-sha256",
      "license": "Example license",
      "license_url": "https://huggingface.co/example/repo/blob/main/LICENSE",
      "revision": "immutable-revision",
      "download_token": "server-signed-opaque-token"
    }
  ]
}
```

### Create jobs

The browser never sends a destination path or download URL when starting a job. It sends only server-signed tokens returned by `inspect` and a mandatory license confirmation.

`POST /server-model-downloader/jobs` with `X-SMD-CSRF`:

```json
{
  "download_tokens": ["server-signed-opaque-token"],
  "license_confirmed": true
}
```

Expected response:

```json
{
  "jobs": []
}
```

### Job status

`GET /server-model-downloader/jobs`

```json
{
  "jobs": [
    {
      "id": "job-id",
      "name": "model.safetensors",
      "directory": "diffusion_models",
      "status": "downloading",
      "bytes_downloaded": 1000000,
      "size": 123456789,
      "progress": 0.0081,
      "error": null,
      "created_at": "2026-08-19T00:00:00Z",
      "updated_at": "2026-08-19T00:00:02Z"
    }
  ]
}
```

Active jobs are polled every two seconds. Both fractional progress (`0..1`) and percentage progress (`0..100`) are accepted.

### Cancel a job

`POST /server-model-downloader/jobs/{job_id}/cancel` with `X-SMD-CSRF` and an empty JSON object.

## Backend integration

The backend package should expose this directory as its `WEB_DIRECTORY`. A typical package layout is:

```text
server_model_downloader/
├── __init__.py
└── frontend/
    ├── server-model-downloader.js
    └── server-model-downloader.css
```

The Python package then exports:

```python
WEB_DIRECTORY = "./frontend"
```

ComfyUI automatically loads JavaScript files from the exported web directory. The CSS file is loaded by the JavaScript using `import.meta.url`, so it remains correct under ComfyUI's generated extension URL.

## Security properties enforced by the frontend

- The browser starts downloads only with opaque, server-signed `download_token` values.
- A license checkbox is mandatory for every submitted batch and resets when selection changes.
- CSRF is obtained from `/session`, sent in `X-SMD-CSRF`, and refreshed once after an explicit `403`.
- Dynamic backend content is inserted with `textContent`; it is never written with `innerHTML`.
- Model and license hyperlinks are rendered only for valid HTTPS URLs.
- Destination paths, revisions, checksums, eligibility, and source safety remain server-authoritative.

Frontend checks are defense in depth only. The backend must still validate tokens, URL hosts and revisions, safe extensions, allowed destination directories, available disk space, redirect targets, download size, checksums, concurrency, authentication, and CSRF.

Promoted subgraph inputs are discovered when the active workflow's root model metadata contains the selected model. If model metadata exists only on an interior promoted-input node, expand that subgraph and verify the model manually; the public extension API does not expose the frontend's internal promoted-widget source resolver.

## Manual verification

1. Open a workflow template containing root workflow model metadata or `node.properties.models` metadata.
2. Open **Extensions → AWS 模型下载 → 下载到 AWS** or click the new sidebar icon.
3. Click **扫描工作流** and verify that each currently selected model appears once, with referencing nodes shown.
4. Confirm installed and ineligible rows cannot be selected.
5. Select an eligible model and verify the submit button remains disabled until the license checkbox is selected.
6. Start a job and verify progress updates without a browser file download.
7. Cancel an active job and verify its state changes.
8. Complete a small test download and confirm loader combo boxes refresh without reloading the page.
