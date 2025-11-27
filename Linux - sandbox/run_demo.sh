#!/usr/bin/env bash
set -e

ROOT="/srv/sandbox-root"

echo "[*] Running demo inside sandbox"
sudo ../py_sandbox.py "$ROOT" /bin/sh -c "id; hostname; ps aux"
