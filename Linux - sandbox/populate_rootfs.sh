#!/usr/bin/env bash
set -euo pipefail

ROOTFS="$1"
BIN="$2"

if [[ -z "$ROOTFS" || -z "$BIN" ]]; then
    echo "Usage: populate_rootfs.sh <rootfs> <binary>"
    exit 1
fi

mkdir -p "$ROOTFS"

# copy the binary
mkdir -p "$ROOTFS$(dirname "$BIN")"
cp -v "$BIN" "$ROOTFS$BIN"

# copy all libs required by ldd
for lib in $(ldd "$BIN" | awk '/=>/ {print $(NF-1)} /^\// {print $1}'); do
    if [[ -f "$lib" ]]; then
        echo "[*] Copying $lib"
        mkdir -p "$ROOTFS$(dirname "$lib")"
        cp -v "$lib" "$ROOTFS$lib"
    fi
done

echo "[*] Done populating rootfs"
