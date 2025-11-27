#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:?Need rootfs path}
METHOD=${2:-busybox}

mkdir -p "$ROOT"

if [[ "$METHOD" == "busybox" ]]; then
    echo "[*] Building BusyBox rootfs at $ROOT"

    sudo mkdir -p "$ROOT"/{bin,lib,lib64,proc,dev,etc,tmp}
    sudo chmod 1777 "$ROOT/tmp"

    BB=$(which busybox)
    if [[ -z "$BB" ]]; then
        echo "BusyBox not installed. Installing now..."
        sudo apt update
        sudo apt install -y busybox
        BB=$(which busybox)
    fi

    sudo cp "$BB" "$ROOT/bin/"

    for app in sh id hostname ps ls cat echo grep sleep uname mount umount; do
        sudo ln -sf /bin/busybox "$ROOT/bin/$app"
    done

    for lib in $(ldd "$BB" | awk '/=>/ {print $(NF-1)} /^\// {print $1}'); do
        sudo mkdir -p "$ROOT$(dirname "$lib")"
        sudo cp -v "$lib" "$ROOT$lib"
    done

    sudo cp -vf /etc/resolv.conf /etc/nsswitch.conf /etc/hosts "$ROOT/etc/" || true

    echo "[*] BusyBox rootfs created."
    exit 0
fi

if [[ "$METHOD" == "debootstrap" ]]; then
    echo "[*] Running debootstrap..."

    sudo apt install -y debootstrap
    sudo debootstrap --arch=amd64 jammy "$ROOT" http://archive.ubuntu.com/ubuntu

    sudo cp -v /etc/resolv.conf "$ROOT/etc/"
    echo "[*] Done."
    exit 0
fi

echo "Unknown method: $METHOD"
exit 1
