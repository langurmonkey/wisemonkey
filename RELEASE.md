# Releasing Wisemonkey

This document describes how to cut a new release of Wisemonkey and publish it to
[PyPI](https://pypi.org/project/wisemonkey/).

## Overview

- **Version source:** `pyproject.toml` (`version = "YYYY.M.D"`). This is the
  single source of truth; it is read at runtime via `importlib.metadata`.
- **Build backend:** hatchling. The wheel packages `agent` and `tools`.
- **Publish credential:** `.env-uv` in the repo root holds
  `UV_PUBLISH_TOKEN=pypi-…`. It is gitignored (matched by `.env*`), and `uv`
  reads `UV_PUBLISH_TOKEN` from the environment automatically.

## Release steps

### 1. Bump the version

Edit `pyproject.toml`:

```toml
version = "2026.9.16"   # set to the new version
```

### 2. Verify before building

```bash
just test          # uv run python -m unittest discover -s tests -v
just checkall      # ty check agent tools tests
```

### 3. Commit and tag

```bash
git add pyproject.toml
git commit -m "Release 2026.9.16"
git tag 2026.09.16
```

### 4. Clean and build

Stale artifacts in `dist/` will otherwise be re-uploaded, so remove them first:

```bash
rm -rf dist/
uv build           # produces dist/wisemonkey-<version>-py3-none-any.whl + .tar.gz (binary: `wmk`)
```

### 5. Publish to PyPI

Load the token, then publish:

```bash
set -a; source .env-uv; set +a
uv publish
```

`uv publish` picks up `UV_PUBLISH_TOKEN` from the environment.

### 6. Push commit and tag

```bash
git push origin master --follow-tags
git push github master --follow-tags   # if you mirror to GitHub
```

## Notes / gotchas

- **Keep the working tree clean.** `uv.lock` is often modified — commit or
  stash it before tagging so the release commit is clean.
- **Version scheme.** CalVer `YYYY.M.D` is used for both tags and the
  `pyproject.toml` version. Keep them consistent; PyPI requires a
  PEP 440-compliant version, and both forms are fine.
- **Dry run on TestPyPI** (optional):

  ```bash
  uv publish --publish-url https://test.pypi.org/legacy/ --token <testpypi-token>
  ```

- **Token security.** `.env-uv` is matched by `.env*` in `.gitignore`, so it
  won't be committed. Rotate the token on PyPI if it is ever exposed.
- **Verify after publishing:**

  ```bash
  pip index versions wisemonkey
  ```

  or check https://pypi.org/project/wisemonkey/.
