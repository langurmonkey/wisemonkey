# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# wisemonkey in a container.
#
# Nothing here needs a BuildKit-only feature, and the `syntax` directive above
# is a comment to the classic builder (it is ignored), so the same file builds
# under either. It does need a real build, though: the first `wmkc` run takes a
# few minutes (apt, uv sync, tiktoken), and the image does not track the
# checkout afterwards -- rebuild after changing code.
#
# The image *is the agent*: code, dependencies and tokenizer data live here, so
# a run needs no host Python and no network. What moves at run time is the
# workspace -- `wmkc` bind-mounts the current directory and the host's
# config/session directories and runs `wmk` as the calling user, so what the
# agent writes stays yours.
#
#   build:  docker build -t wisemonkey-agent:local .
#   run:    ./wmkc        (its mounts and sandbox flags are the point; see there)
# ---------------------------------------------------------------------------

FROM python:3.13-slim

# uv, copied from its official image rather than curl'd: pinned, and no
# installer script runs at build time. Mirrors the host's `uv --version`.
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /usr/local/bin/

# What the agent reaches for. `git` is most of what a coding agent's
# `run_command` does; `ripgrep` backs its own searches; `less`/`nano` are the
# fallback editors for /edit and /config edit when the host exports no $EDITOR;
# without `ca-certificates` every TLS request the agent makes fails.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git less nano ripgrep xz-utils \
    && rm -rf /var/lib/apt/lists/*

# uv must not fetch its own Python (this image is the interpreter), must not
# hardlink out of a cache that later gets shadowed, and should byte-compile
# now: at run time the source tree is root-owned and read-only to the caller.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/wisemonkey/.venv

WORKDIR /opt/wisemonkey

# Dependencies first -- they change far less often than the source, so editing
# code reuses this layer instead of re-resolving the lockfile.
COPY pyproject.toml uv.lock README.md .python-version ./
RUN uv sync --frozen --no-install-project --no-dev

# Then the source, installed editable (uv's default for the root project).
# That is not cosmetic: agent/config.py derives its default config from
# `Path(__file__).parent.parent / "config.yaml"`, which only resolves to the
# repo root while the package is *not* copied into site-packages.
COPY . .
RUN uv sync --frozen --no-dev

# COPY keeps the host's mode bits, and a checkout may well contain 0600 files
# (this one does: agent/agent.py, AGENTS.md, README.md are all user-only). The
# container runs as the *calling user*, not root, so such a file arrives
# root-owned and unreadable -- the agent then dies at import with
# PermissionError: /opt/wisemonkey/agent/agent.py. The virtualenv is created by
# uv under the build umask and is already world-readable, so this is a no-op
# for everything but the copied source.
RUN chmod -R a+rX /opt/wisemonkey

# Bake the tiktoken encodings in, so token counting never needs the network at
# run time (a sandboxed container may have none). Outside every mounted path,
# and world-readable because the container runs as an arbitrary uid.
#
# The interpreter is spelled out in full: the venv's bin/ is only prepended to
# PATH below, and a bare `python` here is the base image's, which has no
# tiktoken installed -- the whole step is a `ModuleNotFoundError` under it.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN mkdir -p "$TIKTOKEN_CACHE_DIR" \
    && "$UV_PROJECT_ENVIRONMENT/bin/python" -c \
        "import tiktoken; [tiktoken.get_encoding(n) for n in ('cl100k_base', 'o200k_base')]" \
    && chmod -R a+rX "$TIKTOKEN_CACHE_DIR"

# Install dangerous command guard (dcg). The build runs as root, so an
# installer that defaults to ~/.local/bin lands the binary in /root/.local/bin,
# which is on nobody's PATH at run time (`wmkc` runs as the calling user and
# forwards the host's HOME, and only ~/.gitconfig is bind-mounted from home).
# So place it on PATH and fail the build if it did not arrive.
RUN curl -fsSL "https://raw.githubusercontent.com/Dicklesworthstone/destructive_command_guard/main/install.sh?$(date +%s)" \
      | bash -s -- --easy-mode \
    && D="$(find /root /usr/local /opt -type f -name dcg -perm -u+x 2>/dev/null | head -1)" \
    && test -n "$D" \
    && install -m 0755 "$D" /usr/local/bin/dcg \
    && dcg --version

# `wmk` (and the deprecated `wisemonkey` alias) on PATH.
ENV PATH="/opt/wisemonkey/.venv/bin:$PATH"

# Deliberately no HOME or XDG_* here: `wmkc` forwards the host's, so the
# container reads the same config.yaml, session history and .env the host does.
# Baking a fixed XDG_CONFIG_HOME would silently hide the mounted configuration.
CMD ["wmk"]
