# Security Policy

## Supported versions

Security fixes are provided for the latest released version. Pre-release branches and modified deployments are supported only on a best-effort basis.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's **Security → Report a vulnerability** private reporting flow for this repository. Include the affected version, configuration, reproduction steps, impact, and any proposed mitigation. Do not include production credentials, JWTs, provider tokens, model download tokens, or private workflow data.

The maintainer will acknowledge a complete report within seven days, coordinate remediation and disclosure privately, and credit reporters who request attribution.

## Deployment requirements

- Keep ComfyUI behind an authenticated HTTPS reverse proxy. Do not expose its listener directly to the public Internet.
- Leave `SMD_ENABLED=false` until an exact HTTPS `SMD_PUBLIC_ORIGIN` and a supported authentication mode are configured.
- In Cloudflare Access mode, protect the entire hostname and do not bypass downloader paths.
- In trusted-proxy mode, accept asserted identities only from minimal proxy CIDRs, and overwrite the identity header at the proxy.
- Run one ComfyUI Python process for each downloader state/model directory pair.
- Keep the state directory persistent, mode `0700`, and owned only by the ComfyUI service account.
- Store provider credentials only in the server process environment.
- Maintain a separate backup of valuable models; the extension intentionally never overwrites an existing destination.

## Security boundaries

The browser may select server-inspected model tokens and request actions, but it is not trusted to choose arbitrary URLs, paths, revisions, sizes, or checksums. The provider, proxy, filesystem, and workflow metadata are also treated as untrusted inputs.

The backend enforces authentication, origin/CSRF checks, provider URL and redirect validation, public-address DNS resolution, fixed destination categories, capacity reservations, safe resumable files, checksums, Safetensors parsing, per-target locks, and no-replace publication. Frontend restrictions are defense in depth only.

The following are deliberately out of scope for version 1:

- Windows and macOS servers.
- Multiple ComfyUI processes sharing downloader state.
- Arbitrary download hosts or file formats.
- Shared-password authentication.
- Automatic writes to `extra_model_paths.yaml` locations.

## Operational response

If compromise is suspected:

1. Set `SMD_ENABLED=false` and restart ComfyUI.
2. Revoke Cloudflare Access sessions and rotate provider credentials.
3. Preserve the private state directory and service logs for investigation.
4. Verify model hashes against the provider before re-enabling the extension.
5. Rotate the state signing key only after active resumable jobs have been accounted for; existing signed tokens will become invalid.
