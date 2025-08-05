
#!/usr/bin/env python3

import sys
import struct
import glob
import shutil
import os
import hashlib
import zlib

def get_data(args):
    fn1, fn2 = args

    with open(fn1, "rb") as f:
        d1 = f.read()
    with open(fn2, "rb") as f:
        d2 = f.read()

    assert d1.startswith(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    assert d2.startswith(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    # make sure the header of the two files match
    assert d1[:0x21] == d2[:0x21]
    # make sure the first 21 bytes of the 2 PNG files are the same
    return d1, d2


d1, d2 = get_data(sys.argv[1:3])

hash = "India"

print("Header hash: %s" % hash)

if not glob.glob("png1-%s.bin" % hash):
    print("Not found! Launching computation...")

    # make the complete prefix
    with open("prefix", "wb") as f:
        f.write(b"".join([
            # 00-20 - original common header
            d1[:0x21],
            # 21-46 - padding chunk
            b"\0\0\0\x1a", b"lUBN", b"ha ri is PRO gamerHEHEHEHE", zlib.crc32(b"aNGE" + b"ha ri is PRO gamerHEHEHEHEE").to_bytes(4, 'big'),
            #The first block ends at 64 and from the 65th byte the second prefix block starts  where the 10th byte of the second prefix block will be differ by +1 becoz of unicoll
            #the first prefix block - 00 to 40
            #the second prefix block is 41-4F (16 bytes long which follows the rule should be less than 20 bytes and should be amultiple of 4)
            # 47-C7 - collision chunk

            # 47-4F
            # this is the end of the collision prefix,
            #So in the 74th byte which is 64 +10 should be the length of the next chunk which will differ by 1 which should be 171 in the second collision block
            # => lengths of 0x71 and 0x171
            # ")" is the start of the collision and the chunk length is chosen to be 0x71== 133 in decimal for the reason the unicoll produces a collision block 
            # of 192 bytes where 64 bytes is the prefix and 16 bytes are appended to make the fastcoll work,(192-80=113 bytes needed by another chunk to fill which is why the length is 0x71)
            b"\0\0\0\x71", b"dORA", b")",

            # the end of the collision blocks if they're not computed
            # 50-BF
            # " " * 0x70,
        ]))

    os.system("../hashclash/scripts/poc_no.sh prefix")
    
    shutil.copyfile("collision1.bin", "png1-%s.bin" % hash)
    shutil.copyfile("collision2.bin", "png2-%s.bin" % hash)

with open("png1-%s.bin" % hash, "rb") as f:
    block1 = f.read()
with open("png2-%s.bin" % hash, "rb") as f:
    block2 = f.read()

assert len(block1) == 0xC0
assert len(block2) == 0xC0
# make sure both the block lengths are 192 bytes
ascii_art = b":)" * 122
assert len(ascii_art) == 0xF4
# assert the length of the ascii art is 244 bytes
suffix = b"".join([

    # The fake CRC for the chunk dORA
    b"EHaL",
    # the remaining of the chunk mARC

    # since the difference in the first and the second png is of 256 bytes (0x100) of the chunk dORA,So we fill that with an ascii art.
    #For the first png file the dORA chunk ends at ----- so there must be a next fake chunk after which the actual png1 file info is appended,
    #Whereas for the second png the dORA chunk is extra 256 bytes long so the jump chunk will be part of the data in the dORA's chunk itself
    # The 256 bytes are filled by (4(chunk length)+4(chunk name)+244(chunk data)+(chunk length)).Therefore the second png data will be read.

    # the length, the type and the data should all take 0x100
    struct.pack(">I", 0x100 - 4 * 2 + len(d2[0x21:])),
    b"jUMP",
    # it will cover the data chunks of d2,
    # and the 0x100 buffer
    ascii_art,
    b"\xDE\xAD\xBE\xEF",
    # fake cyclic redundancy check for mARC

    # 1C8 - Img2 + 4
    d2[0x21:],
    b"\x5E\xAF\x00\x0D",
    # fake cyclic redundancy check for jUMP after d2's IEND
    d1[0x21:],
])

with open("%s-1.png" % hash, "wb") as f:
    f.write(b"".join([
        block1,
        suffix
    ]))

with open("%s-2.png" % hash, "wb") as f:
    f.write(b"".join([
        block2,
        suffix
    ]))
