#!/bin/sh
# Populate predownloaded/ with every tool needed to build panfrost_vk_compile
# on an airgapped Debian 12 (bookworm) sandbox.  Run this on a CONNECTED
# Debian 12 machine (bookworm) so apt resolves the right .debs.
#
# bookworm's default repos only ship LLVM 14, so this script adds the official
# LLVM apt repo (apt.llvm.org) to obtain llvm-15 / libclang-15 / libclc-15.
set -eu

cd "$(dirname "$0")"
ROOT=$(pwd)
DEBS="$ROOT/debs"
WHEELS="$ROOT/wheels"
BIN="$ROOT/bin"
BINDGEN_VERSION=0.71.1
LLVM_VER=15
DISTRO_CODENAME=bookworm

mkdir -p "$DEBS" "$WHEELS" "$BIN"

# --- Sanity: we target Debian 12 (bookworm) --------------------------------
# Running on another distro (e.g. Ubuntu) would resolve the WRONG packages and
# add the bookworm LLVM repo to the host's apt.  Use fetch-bookworm.sh (a
# container wrapper) instead, or set ALLOW_NON_BOOKWORM=1 to force.
if [ -r /etc/os-release ]; then
  . /etc/os-release
  if [ "${VERSION_CODENAME:-}" != "$DISTRO_CODENAME" ] && [ "${ALLOW_NON_BOOKWORM:-0}" != "1" ]; then
    echo "ERROR: this script targets Debian $DISTRO_CODENAME; detected ${PRETTY_NAME:-unknown}." >&2
    echo "       Running it here would fetch the wrong .debs and add the bookworm" >&2
    echo "       LLVM repo to this host.  Use ./fetch-bookworm.sh (container), or" >&2
    echo "       set ALLOW_NON_BOOKWORM=1 to force." >&2
    exit 1
  fi
fi

# --- Add the LLVM apt repo (bookworm has no llvm-15 in default repos) ------
if ! apt-cache show "llvm-${LLVM_VER}-dev" >/dev/null 2>&1; then
  echo "==> Adding LLVM ${LLVM_VER} apt repo for ${DISTRO_CODENAME}"
  apt-get install -y --no-install-recommends ca-certificates curl gnupg
  curl -fsSL "https://apt.llvm.org/llvm-snapshot.gpg.key" \
    | gpg --dearmor -o /usr/share/keyrings/llvm.gpg
  echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/${DISTRO_CODENAME}/ llvm-toolchain-${DISTRO_CODENAME}-${LLVM_VER} main" \
    > "/etc/apt/sources.list.d/llvm-${LLVM_VER}.list"
fi

echo "==> apt-get update"
apt-get update

# --- Fetch .deb packages (with the full dependency closure) ----------------
echo "==> Fetching .deb packages (with dependencies)"
PKGS="build-essential pkg-config bison flex libdrm-dev libelf-dev \
  llvm-${LLVM_VER}-dev libclang-${LLVM_VER}-dev libclc-${LLVM_VER}-dev \
  libllvmspirvlib-${LLVM_VER}-dev spirv-tools spirv-headers glslang-tools"
# --download-only resolves the full dependency closure into the apt cache.
# shellcheck disable=SC2086
apt-get install -y --download-only $PKGS || {
  echo "apt-get download-only failed; falling back to per-package download" >&2
  for p in $PKGS; do apt-get download "$p" || true; done
}
# Copy the freshly downloaded archives out of the apt cache.
cp -f /var/cache/apt/archives/*.deb "$DEBS"/ 2>/dev/null || true
# Also pick up anything downloaded into the cwd by the fallback path.
mv -f ./*.deb "$DEBS"/ 2>/dev/null || true

# --- Build SPIRV-Tools >= 2024.1 (bookworm ships 2023.1) -------------------
# Mesa 27's CLC path requires SPIRV-Tools >= 2024.1; build it from source and
# drop the .deb into debs/.  Non-fatal: the build will fail at configure if it
# is missing, but the rest of the bundle is still produced.
if [ -x "$ROOT/build-spirv-tools.sh" ]; then
  "$ROOT/build-spirv-tools.sh" || \
    echo "WARNING: SPIRV-Tools build failed; the CLC build will need it." >&2
fi

# --- Fetch Python wheels (bookworm python 3.11 compatible) -----------------
echo "==> Fetching Python wheels"
python3 -m pip download --dest "$WHEELS" \
  meson ninja mako packaging PyYAML

# --- Fetch bindgen (prebuilt; optional for the non-rust build) -------------
echo "==> Fetching bindgen $BINDGEN_VERSION (prebuilt)"
ok=0
for target in x86_64-unknown-linux-gnu x86_64-unknown-linux-musl; do
  url="https://github.com/rust-lang/rust-bindgen/releases/download/v${BINDGEN_VERSION}/bindgen-${BINDGEN_VERSION}-${target}.tar.gz"
  if curl -fsSL "$url" -o /tmp/bindgen.tar.gz; then
    tar -xzf /tmp/bindgen.tar.gz -C /tmp
    # The archive contains a single `bindgen` binary at its root.
    find /tmp -maxdepth 2 -name bindgen -type f -exec cp -f {} "$BIN"/ \;
    ok=1
    break
  fi
done
if [ "$ok" -ne 1 ]; then
  # bindgen is only needed for the rust-kraid variant; do not fail the whole
  # fetch if it is unavailable (the default non-rust build does not need it).
  echo "WARNING: could not fetch bindgen; skipping (rust-kraid variant only)." >&2
else
  chmod +x "$BIN"/bindgen 2>/dev/null || true
fi

echo
echo "==> Done. Contents:"
du -sh "$DEBS" "$WHEELS" "$BIN" 2>/dev/null || true
echo "Copy the whole predownloaded/ folder into the sandbox and run ./install.sh"