#!/bin/sh
# Build SPIRV-Tools >= 2024.1 from source and package it as a .deb in debs/.
# Debian 12 (bookworm) only ships SPIRV-Tools 2023.1, but Mesa 27's CLC path
# requires >= 2024.1.  Run inside the bookworm container (or on bookworm).
#
# Uses SPIRV-Tools' own utils/git-sync-deps to fetch the matching SPIRV-Headers
# (and other deps), so we don't hand-manage the dependency versions.
set -eu

cd "$(dirname "$0")"
ROOT=$(pwd)
DEBS="$ROOT/debs"
# The v2024.1 tree ref is not a branch; pin the exact commit.
SPIRV_TOOLS_COMMIT=04896c462d9f3f504c99a4698605b6524af813c1
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "==> Building SPIRV-Tools $SPIRV_TOOLS_COMMIT from source"
apt-get install -y --no-install-recommends \
  cmake ninja-build git build-essential python3

git clone --filter=blob:none https://github.com/KhronosGroup/SPIRV-Tools "$WORK/SPIRV-Tools"
git -C "$WORK/SPIRV-Tools" checkout "$SPIRV_TOOLS_COMMIT"
# Fetch the matching SPIRV-Headers and other deps into external/.
( cd "$WORK/SPIRV-Tools" && python3 utils/git-sync-deps )

cmake -S "$WORK/SPIRV-Tools" -B "$WORK/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSPIRV_BUILD_FUZZER=OFF \
  -DSPIRV_SKIP_TESTS=ON \
  -DCMAKE_INSTALL_PREFIX=/usr
cmake --build "$WORK/build" --parallel "$(nproc)"
DESTDIR="$WORK/stage" cmake --install "$WORK/build"

STAGE="$WORK/stage"
mkdir -p "$STAGE/DEBIAN"
cat > "$STAGE/DEBIAN/control" <<EOF
Package: spirv-tools-2024
Version: 2024.1
Architecture: amd64
Maintainer: predownloaded
Description: SPIRV-Tools 2024.1 built from source (bookworm ships 2023.1)
EOF
dpkg-deb --build "$STAGE" "$DEBS/spirv-tools-2024_2024.1_amd64.deb"
echo "==> Wrote $DEBS/spirv-tools-2024_2024.1_amd64.deb"