# Contributing

Thank you for improving Server Model Downloader.

## Before opening a change

- Use a GitHub issue for behavior changes that affect the public API or security model.
- Report vulnerabilities privately as described in [`SECURITY.md`](SECURITY.md).
- Keep the extension Linux-only, fail-closed, and compatible with one ComfyUI process.
- Do not add arbitrary hosts, unsafe model formats, fallback authentication, or a no-lock publication path.
- Never add real credentials, JWTs, signed download tokens, production paths, or private workflows to tests or logs.

## Development setup

Use Python 3.10–3.13 and Node 20 or newer:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
npm test
npm run check
.venv/bin/python -m pytest --cov=backend --cov-branch backend/tests
.venv/bin/ruff check backend
.venv/bin/ruff format --check backend
```

Tests must use temporary directories and local fake HTTP services. They must not access real provider files or require external credentials.

## Pull requests

- Branch from `main` and use focused commits.
- Add tests for every changed behavior and failure mode.
- Document public API, configuration, compatibility, or security changes.
- Preserve existing JSON fields and routes unless a major release explicitly removes them.
- Run the complete local check suite before requesting review.
- Resolve all review conversations; `main` accepts squash merges only.

By submitting a contribution, you agree that it is licensed under Apache-2.0 as described by the repository license.

## Release checklist

1. Update `CHANGELOG.md` and the version in `pyproject.toml` and `package.json`.
2. Confirm CI and CodeQL pass on the release commit.
3. Confirm the `release` environment contains `REGISTRY_ACCESS_TOKEN` and has appropriate reviewers.
4. Create an annotated `vX.Y.Z` tag from `main`.
5. Let the release workflow build a deterministic source archive, publish checksums and a GitHub Release, and publish the immutable Registry version.
6. Test `comfy node install server-model-downloader` in a clean ComfyUI environment.
