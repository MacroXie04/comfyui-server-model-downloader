# Frontend contract

The browser extension registers the **Server Model Downloader** sidebar and Extensions-menu command for ComfyUI frontend 1.48.7 or newer.

## Behavior

- Opening the sidebar fetches only the authenticated session and job history.
- Provider metadata is requested only after the user chooses **Scan workflow**.
- Scans read model metadata matching active loader widget selections; disabled and bypassed nodes are ignored.
- Every request family uses an `AbortController` plus a generation check, so a closed panel or newer request cannot be overwritten by a stale response.
- Dynamic content is inserted through `textContent`; only validated HTTPS links are rendered.
- The browser submits identity-bound opaque download tokens, never arbitrary destination paths or resolved download URLs.
- License confirmation resets whenever the model selection changes.
- Completed jobs trigger `app.refreshComboInNodes()` without an automatic provider rescan.

## API expectations

Both `/server-model-downloader/*` and `/api/server-model-downloader/*` expose the same authenticated contract. The frontend uses the first prefix.

`GET /session` returns a CSRF token and expiry, API and extension versions, allowed model directories, safe extensions, capabilities, and minimal `{email, auth_mode}` identity data.

`POST /inspect` accepts no more than 50 workflow records with `name`, canonical provider `url`, and allowlisted `directory`. It returns server-resolved metadata and an opaque `download_token` for each eligible model.

`POST /jobs` accepts only `download_tokens` and `license_confirmed: true`. `GET /jobs?limit=50&cursor=…` returns newest-first `{jobs, next_cursor}` pages. Legacy unwrapped job arrays are still accepted by the client.

Jobs retain the existing fields and add:

```json
{
  "phase": "retrying",
  "error_code": "upstream_rate_limited",
  "attempt": 2,
  "max_attempts": 5,
  "cancel_requested": false
}
```

Public phases are `queued`, `connecting`, `downloading`, `retrying`, `hashing`, `validating`, `publishing`, `cancelling`, `completed`, `failed`, and `cancelled`.

`POST /jobs/{id}/cancel` requests cancellation. `DELETE /jobs/{id}/partial` removes retained data only for a terminal failed or cancelled job. Both require `X-SMD-CSRF` and the configured HTTPS origin.

Error responses use `{error, code, request_id}`. The UI may display the request ID for support, but must never receive or render JWTs, provider tokens, signed download tokens, absolute paths, or upstream response bodies.

See [`../docs/openapi.yaml`](../docs/openapi.yaml) for the machine-readable contract.

## Files

- `server-model-downloader.js` registers and renders the sidebar.
- `server-model-downloader.css` provides dependency-free ComfyUI-aware styling.
- `model-scan.mjs` extracts the active workflow's selected model metadata.
- `client-state.mjs` normalizes API data and prevents stale asynchronous updates.

## Checks

```bash
npm test
npm run check
```

The frontend tests cover graph traversal, inactive nodes, pagination, additive job fields, progress compatibility, safe external links, request cancellation/generation behavior, English branding, explicit scan semantics, and accessible progress metadata.
