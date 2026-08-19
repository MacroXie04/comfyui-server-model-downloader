# ComfyUI Server Model Downloader

A restricted ComfyUI extension that downloads missing workflow models directly to the ComfyUI server instead of the browser.

It adds a **Download to Server** sidebar and Extensions-menu entry, scans the active workflow for selected model metadata, resolves provider metadata, and runs verified resumable downloads on the server.

This is a private/internal repository. Installation and use are limited to the copyright holder and people who have received explicit permission.

## Security model

This extension is intended for a single trusted administrator behind an authenticated access layer such as Cloudflare Access. It does **not** provide user authentication by itself. Keep ComfyUI bound to loopback or a private network and do not expose its HTTP service directly to the public Internet.

Before enabling the extension, verify all of the following:

- The entire ComfyUI hostname, including `/server-model-downloader/*`, is covered by a deny-by-default Access application with no public bypass policy.
- ComfyUI listens only on `127.0.0.1` (or another explicitly private origin interface).
- The cloud firewall/security group has no public inbound rule for the ComfyUI port.
- An unauthenticated request to `https://comfy.example.com/server-model-downloader/session` is redirected to the identity provider or denied; it must never return downloader JSON.
- The reverse proxy overwrites `Host`/`X-Forwarded-Host` with trusted values. Do not pass a client-supplied `X-Forwarded-Host` through unchanged.

Mutating requests require an HTTPS browser origin matching the effective host. The download page therefore intentionally rejects plain `http://127.0.0.1` SSH tunnels and plain HTTP Tailscale URLs. Use the authenticated HTTPS hostname (or provide an equivalently trusted HTTPS reverse proxy).

The backend enforces the security boundary; frontend checks are only defense in depth:

- HTTPS sources are limited to canonical Hugging Face and Civitai model URLs.
- Only `.safetensors` files are accepted.
- Destination directories come from a fixed ComfyUI model-directory allowlist.
- The browser submits short-lived, server-signed download tokens rather than arbitrary URLs or paths.
- Metadata requests and every redirect are revalidated, including public-IP DNS checks to reduce SSRF risk.
- Downloads reserve 20 GiB of free space and enforce provider-reported size limits.
- Existing destination files are never overwritten.
- Partial files resume using HTTP ranges after a service restart.
- SHA-256 and Safetensors structure are checked before descriptor-bound, no-replace publication.
- Per-target filesystem locks prevent concurrent workers from publishing the same model.
- Mutating API calls require same-origin and CSRF validation.

## Compatibility

The deployed reference environment is:

- Ubuntu 24.04
- Python 3.12
- ComfyUI `v0.33.1`
- ComfyUI frontend `v1.48.7`
- ComfyUI-Manager `v4.2.2`

The publication path uses Linux filesystem facilities (`flock`, `linkat`, `/proc/self/fd`) and is therefore Linux-only.

## Installation

Clone the repository into the ComfyUI custom-nodes directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone git@github.com:MacroXie04/comfyui-server-model-downloader.git
```

Install the small runtime dependency set into the same Python environment ComfyUI uses, then restart ComfyUI:

```bash
/path/to/ComfyUI/.venv/bin/python -m pip install -r comfyui-server-model-downloader/requirements.txt
```

By default, models and downloader state are stored at:

```text
/srv/comfyui-data/models
/srv/comfyui-data/user/server-model-downloader
```

Create the persistent directories for the account that runs ComfyUI. Replace `comfyui:comfyui` when your service uses a different account:

```bash
sudo install -d -o comfyui -g comfyui -m 0750 \
  /srv/comfyui-data/models \
  /srv/comfyui-data/user/server-model-downloader
```

Override the locations in the ComfyUI service environment when your installation uses different paths:

```text
SMD_MODELS_ROOT=/path/to/ComfyUI/models
SMD_STATE_DIR=/path/to/private/persistent/state
```

For a systemd service, place the values in an override such as `/etc/systemd/system/comfyui.service.d/server-model-downloader.conf`:

```ini
[Service]
Environment="SMD_MODELS_ROOT=/srv/comfyui-data/models"
Environment="SMD_STATE_DIR=/srv/comfyui-data/user/server-model-downloader"
```

Then reload and restart the service:

```bash
sudo systemctl daemon-reload
sudo systemctl restart comfyui
```

For a manually started ComfyUI process, export the same variables in the shell before starting it.

`SMD_STATE_DIR` contains the signing key and resumable job state. It must be persistent, writable only by the ComfyUI service account, and excluded from backups or repositories that are shared publicly.

Optional provider credentials can be supplied to the ComfyUI service process:

```text
HF_TOKEN=...
HUGGING_FACE_HUB_TOKEN=...  # accepted as an alternative to HF_TOKEN
CIVITAI_API_TOKEN=...
```

Do not put credentials in workflow URLs, Git, or browser storage.

## Post-install verification

Confirm that ComfyUI loaded the browser extension and that the job list is initially available:

```bash
curl -fsS http://127.0.0.1:8188/extensions | grep comfyui_server_model_downloader
curl -fsS http://127.0.0.1:8188/server-model-downloader/jobs
```

Check the service log for import or permission errors:

```bash
sudo journalctl -u comfyui -n 200 --no-pager
```

From a client that is not logged in to the Access application, verify the public endpoint is challenged rather than returning JSON:

```bash
curl -I https://comfy.example.com/server-model-downloader/session
```

Finally, inspect the listener and cloud firewall separately. A safe reference deployment has ComfyUI on `127.0.0.1:8188` and no inbound cloud-firewall rule for port 8188.

## Usage

1. Open a workflow containing model metadata.
2. Open **Extensions → AWS Model Download → Download to AWS**, or select the downloader sidebar.
3. Choose **Scan workflow**.
4. Review provider, immutable revision, destination, file size, SHA-256, and license.
5. Select eligible files, confirm the license, and start the download.
6. Monitor or cancel jobs in the same panel.

ComfyUI's built-in missing-model buttons still download through the browser. Use this extension's panel when the file should be written directly to the server.

Promoted subgraph inputs are detected when the active workflow exposes their model metadata at the root. If metadata exists only on an interior promoted-input node, expand the subgraph and verify the model manually.

## Supported destinations

The server can resolve and write only these model directories:

```text
checkpoints
diffusion_models
text_encoders
clip
clip_vision
vae
loras
controlnet
upscale_models
embeddings
audio_encoders
style_models
```

## Development checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q backend/tests
ruff check backend
node --test frontend/tests/model-scan.test.mjs
node --check frontend/server-model-downloader.js
node --check frontend/model-scan.mjs
```

The backend and API design are documented in more detail in [`frontend/README.md`](frontend/README.md).

## Operational boundary

The current job store and aggregate capacity accounting assume one ComfyUI Python process. Do not run multiple ComfyUI processes against the same model and state directories without adding a process-wide job-store and capacity lock.

## License

Copyright © 2026 Hongzhe Xie. All rights reserved. See [`LICENSE`](LICENSE).
