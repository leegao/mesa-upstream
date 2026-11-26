#!/bin/sh
# Install everything from predownloaded/ into an airgapped Debian sandbox.
# Run as root (or via sudo) INSIDE the sandbox.
set -eu

cd "$(dirname "$0")"
ROOT=$(pwd)

echo "==> Installing .deb packages"
if ls "$ROOT"/debs/*.deb >/dev/null 2>&1; then
  # dpkg -i does not resolve ordering; a second pass fixes any ordering issues.
  dpkg -i "$ROOT"/debs/*.deb || {
    echo "retrying with apt to fix ordering..." >&2
    apt-get install -y --no-download "$ROOT"/debs/*.deb || true
  }
else
  echo "no .debs found in $ROOT/debs (did you run fetch.sh?)" >&2
fi

echo "==> Installing Python wheels (meson, ninja, mako, ...)"
if ls "$ROOT"/wheels/*.whl >/dev/null 2>&1; then
  python3 -m pip install --break-system-packages --no-index --find-links "$ROOT/wheels" \
    meson ninja mako packaging PyYAML
else
  echo "no wheels found in $ROOT/wheels" >&2
fi

echo "==> Installing bindgen"
if [ -f "$ROOT/bin/bindgen" ]; then
  install -m 0755 "$ROOT/bin/bindgen" /usr/local/bin/bindgen
else
  echo "no bindgen binary in $ROOT/bin (optional for the non-rust build)" >&2
fi

echo
echo "==> Verifying"
for t in meson ninja bindgen glslangValidator llvm-config-15; do
  if command -v "$t" >/dev/null 2>&1; then
    printf '  %-16s %s\n' "$t" "$(command -v "$t")"
  else
    printf '  %-16s MISSING (optional?)\n' "$t"
  fi
done
echo "Done. You can now configure and build panfrost_vk_compile."