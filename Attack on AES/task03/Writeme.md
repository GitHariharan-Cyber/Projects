Since the zip is encrypted with 96 bit xor key,we got info about the first ten bytes of the of the zip file header which is 

signature = b'\x50\x4B\x03\x04'      # ZIP Signature - 4 bytes
version = b'\x0A\x00'                # ZIP Version - 2 bytes 
flags = b'\x00\x00\x00\x00'                  # ZIP Flags - 2 bytes 
 
With the first 10 bytes known,for the two unkown bytes we pad up by bruteforcing all the possibilities and check against the zip to open the file.
