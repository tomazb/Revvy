#!/usr/bin/env bash
# build.sh - Build and package the Revvy VS Code extension
# Usage: ./build.sh
set -e

# ── Config ────────────────────────────────────────────────────────────────────
PACKAGE_JSON="package.json"
OUT_DIR="out"

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[build]${NC} $1"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $1"; }
fail() { echo -e "${RED}[error]${NC} $1"; exit 1; }

# ── Checks ────────────────────────────────────────────────────────────────────
command -v node  >/dev/null 2>&1 || fail "node is not installed"
command -v npm   >/dev/null 2>&1 || fail "npm is not installed"

[ -f "$PACKAGE_JSON" ] || fail "Run this script from the extension root directory"

# ── Resolve the vsce runner (Node-version aware) ──────────────────────────────
# vsce is always run through `npx --yes`, never a global install, for two reasons:
#   1. A global bin dir is frequently not on the shell's PATH on Linux
#      ("vsce: command not found" even right after `npm install -g`).
#   2. A globally installed vsce 3.x bundles a newer undici that references the
#      `File` global, which only exists in Node 20+. On Node 18 that throws
#      "ReferenceError: File is not defined" at require time.
# So we pin the vsce major to one compatible with the running Node: 2.x on
# Node < 20, latest otherwise. npx caches the download after the first run.
NODE_MAJOR=$(node -p "process.versions.node.split('.')[0]")
if [ "$NODE_MAJOR" -ge 20 ]; then
  VSCE_PKG="@vscode/vsce"
else
  warn "Node ${NODE_MAJOR} detected (<20) — pinning vsce 2.x (3.x needs Node 20+)"
  VSCE_PKG="@vscode/vsce@2"
fi
VSCE=(npx --yes "$VSCE_PKG")

# ── Bump patch version in package.json ───────────────────────────────────────
log "Bumping patch version..."
npm version patch --no-git-tag-version > /dev/null

# ── Read new version ──────────────────────────────────────────────────────────
VERSION=$(node -p "require('./package.json').version")
NAME=$(node -p "require('./package.json').name")
VSIX_FILE="${NAME}-${VERSION}.vsix"

log "Building ${NAME} v${VERSION}"

# ── Clean previous output ─────────────────────────────────────────────────────
log "Cleaning ${OUT_DIR}/"
rm -rf "$OUT_DIR"

# ── Install dependencies ──────────────────────────────────────────────────────
log "Installing dependencies..."
npm install

# ── TypeScript compile ────────────────────────────────────────────────────────
log "Compiling TypeScript..."
npm run compile

# ── Node 18 `File` polyfill ───────────────────────────────────────────────────
# vsce's transitive deps (@azure/identity → undici) reference the global `File`
# at module-load time. `File` is only a global in Node 20+, so on Node 18 vsce
# crashes with "ReferenceError: File is not defined" before it does any work.
# Node 18.13+ ships `File` in node:buffer, so we expose it on globalThis via a
# small --require preload, scoped (through NODE_OPTIONS) to the vsce process.
NODE_OPTS_FOR_VSCE=""
if [ "$NODE_MAJOR" -lt 20 ]; then
  POLYFILL="$(mktemp "${TMPDIR:-/tmp}/vsce-file-polyfill.XXXXXX.cjs")"
  trap 'rm -f "$POLYFILL"' EXIT
  cat > "$POLYFILL" <<'EOF'
try {
  const buf = require('node:buffer');
  if (typeof globalThis.File === 'undefined' && buf.File) { globalThis.File = buf.File; }
  if (typeof globalThis.Blob === 'undefined' && buf.Blob) { globalThis.Blob = buf.Blob; }
} catch (_) { /* best effort */ }
EOF
  NODE_OPTS_FOR_VSCE="--require ${POLYFILL}"
fi

# ── Package VSIX ─────────────────────────────────────────────────────────────
log "Packaging VSIX..."
NODE_OPTIONS="${NODE_OPTS_FOR_VSCE} ${NODE_OPTIONS:-}" "${VSCE[@]}" package --no-git-tag-version --out "$VSIX_FILE"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
log "Done! Output: ${VSIX_FILE}"
echo ""
echo "  Install locally:"
echo "    code --install-extension ${VSIX_FILE}"
echo ""
