#!/usr/bin/env bash
set -euo pipefail

BINARY_NAME="polymede"
RELEASE_DIR="target/release"
BINARY_PATH="${RELEASE_DIR}/${BINARY_NAME}"

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux)   INSTALL_DIR="/usr/local/bin";;
    FreeBSD) INSTALL_DIR="/usr/local/bin";;
    Darwin)  # macOS: prefer Homebrew prefix if present, else /usr/local/bin
        if command -v brew &>/dev/null && [ -d "$(brew --prefix)/bin" ]; then
            INSTALL_DIR="$(brew --prefix)/bin"
        else
            INSTALL_DIR="/usr/local/bin"
        fi
        ;;
    *)       echo "Unsupported OS: ${OS}" >&2; exit 1;;
esac

# Fallback to user-local if /usr/local/bin is not writable
if [ ! -w "${INSTALL_DIR}" ] 2>/dev/null; then
    INSTALL_DIR="${HOME}/.local/bin"
    mkdir -p "${INSTALL_DIR}"
fi

echo "Detected OS: ${OS}"
echo "Install dir: ${INSTALL_DIR}"

# Build
echo "Building release binary..."
cargo build --release

if [ ! -f "${BINARY_PATH}" ]; then
    echo "Build failed: ${BINARY_PATH} not found" >&2
    exit 1
fi

# Install (symlink)
TARGET="${INSTALL_DIR}/${BINARY_NAME}"
rm -f "${TARGET}"
ln -s "$(realpath "${BINARY_PATH}")" "${TARGET}"

echo ""
echo "Installed: ${TARGET}"
echo "Run '${BINARY_NAME}' to start."
