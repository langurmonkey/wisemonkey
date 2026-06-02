#!/usr/bin/env bash
# Wisemonkey installer
# Usage: curl -fsSL https://codeberg.org/langurmonkey/wisemonkey/raw/branch/master/install.sh | bash

set -euo pipefail

# Check for uv
if ! command -v uv &>/dev/null; then
    echo "Error: uv is required but not found. Install it from https://github.com/astral-sh/uv"
    exit 1
fi

# Configuration
REPO="wisemonkey"
OWNER="langurmonkey"
BRANCH="${BRANCH:-master}"
REPO_URL="https://codeberg.org/$OWNER/$REPO.git"

# XDG-compliant paths with macOS support
if [ "$(uname)" = "Darwin" ]; then
    export XDG_DATA_HOME="${XDG_DATA_HOME:-$HOME/Library}"
    export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/Library/Preferences}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/Library/Caches}"
fi

XDG_DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
INSTALL_DIR="${INSTALL_DIR:-$XDG_DATA/wisemonkey/repository}"

# Detect platform for symlink location
if [ "$(uname)" = "Darwin" ]; then
    BIN_DIR="$HOME/.local/bin"
else
    BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
fi

# Clone or update the repository
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "Updating existing installation..."
    cd "$INSTALL_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo "Cloning wisemonkey repository..."
    mkdir -p "$(dirname "$INSTALL_DIR")"
    git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$INSTALL_DIR"
fi

# Create venv and install with uv
echo "Installing wisemonkey with uv..."
cd "$INSTALL_DIR"
if [ ! -d ".venv" ]; then
    uv venv
fi
uv sync

# Create default config if not exists
XDG_CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}"
CONFIG_DIR="$XDG_CONFIG/wisemonkey"
if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
    mkdir -p "$CONFIG_DIR"
    echo "Creating default config at $CONFIG_DIR/config.yaml"
    cp "$INSTALL_DIR/config.yaml" "$CONFIG_DIR/config.yaml"
fi

# Create wrapper script in ~/.local/bin
mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/wisemonkey" << WRAPPER
#!/bin/bash
INSTALL_DIR="$INSTALL_DIR"
if [ ! -d "\$INSTALL_DIR" ] || [ ! -f "\$INSTALL_DIR/pyproject.toml" ]; then
    echo "Error: Could not find wisemonkey installation"
    exit 1
fi
exec uv --project "\$INSTALL_DIR/pyproject.toml" run wisemonkey "\$@"
WRAPPER
chmod +x "$BIN_DIR/wisemonkey"
echo "Created wrapper: $BIN_DIR/wisemonkey"

# Ensure ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo ""
    echo "⚠️  $BIN_DIR is not in your PATH. Add it to your shell config:"
    echo "  export PATH=\"$BIN_DIR:\$PATH\""
fi

# Show next steps
echo ""
echo "✅ wisemonkey installed successfully!"
echo ""
echo "Next steps:"
echo "  1. Edit config: wisemonkey --edit-config"
echo "  2. Run: wisemonkey"
echo "  3. Update: wisemonkey --update"
