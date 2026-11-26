#!/bin/sh
# Fetch bookworm .debs/wheels/binaries from a NON-bookworm host (e.g. Ubuntu) by
# running fetch.sh inside a Debian 12 (bookworm) container.  This guarantees the
# downloaded packages are the bookworm ones the airgapped sandbox expects.
#
# Requires docker or podman.  The predownloaded/ folder is bind-mounted into the
# container, so debs/ wheels/ bin/ are populated on the host.
set -eu

cd "$(dirname "$0")"
ROOT=$(pwd)

if command -v docker >/dev/null 2>&1; then
  ENGINE=docker
elif command -v podman >/dev/null 2>&1; then
  ENGINE=podman
else
  echo "ERROR: need docker or podman to run the bookworm container." >&2
  exit 1
fi

echo "==> Using $ENGINE to run fetch.sh inside debian:bookworm"
# --network is the default (bridge); the container needs network for apt/pip/curl.
"$ENGINE" run --rm \
  -v "$ROOT":/predownloaded \
  -w /predownloaded \
  debian:bookworm \
  sh -c '
    set -e
    apt-get update
    apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg python3 python3-pip tar
    ./fetch.sh
  '

echo
echo "==> Done. debs/ wheels/ bin/ now contain Debian bookworm packages."
echo "Copy the whole predownloaded/ folder into the airgapped bookworm sandbox."