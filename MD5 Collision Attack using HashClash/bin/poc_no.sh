#!/bin/bash

# Check if prefix file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <prefix_file>"
  exit 1
fi

PREFIX_FILE="$1"

if [ ! -f "$PREFIX_FILE" ]; then
  echo "Error: Prefix file '$PREFIX_FILE' not found!"
  exit 1
fi

# Ensure md5_fastcoll is available
if ! command -v md5_fastcoll &> /dev/null; then
  echo "Error: md5_fastcoll not found in PATH."
  exit 1
fi

# Ensure the prefix is exactly 64 bytes
truncate -s 64 "$PREFIX_FILE"

# Generate two colliding MD5 blocks
md5_fastcoll -o temp1.bin temp2.bin

# Prepend the 64-byte prefix to both collision blocks
cat "$PREFIX_FILE" temp1.bin > collision1.bin
cat "$PREFIX_FILE" temp2.bin > collision2.bin

# Clean up
rm temp1.bin temp2.bin

echo "collision1.bin and collision2.bin created with 192-byte length."
