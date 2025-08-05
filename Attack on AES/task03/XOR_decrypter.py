import math
import io 
from zipfile import ZipFile, BadZipfile
import datetime


def xor_bytes(encrypted_data,final_key):
    # xor the current integers
    return bytes(b1 ^ b2 for b1,b2 in zip(encrypted_data,final_key))
                 
def try_to_decrypt(encrypted_data,predicted_key):
    # here we get the key up to the length of the cypherbytes and
    # xor the cypherbytes and the key at last, which we return
    length_crypt_data = len(encrypted_data)
    length_predicted_key = len(predicted_key)
    key_to_be_repeated = (length_crypt_data//length_predicted_key)
    # we have to find the remainder of the key
    remainder = length_crypt_data%length_predicted_key
    final_key = predicted_key*key_to_be_repeated + predicted_key[:remainder]
    return xor_bytes(encrypted_data,final_key)


def is_zipfile(input_bytes: bytes, key_to_try: bytes) -> bool:
    try:
        """Tries to open the zipfile with the decryptedbytes"""
        with ZipFile(io.BytesIO(input_bytes)) as archive:
            print()
            print(f"key {key_to_try} leads to a successfull decryption")
            print()
            print("Now printing contents of archive...")
            print()
            for info in archive.infolist():
                print(f"Filename: {info.filename}")
                print(f"Modified: {datetime.datetime(*info.date_time)}")
                print(f"Normal size: {info.file_size} bytes")
                print(f"Compressed size: {info.compress_size} bytes")
                print("-" * 20)

            for filename in archive.namelist():
                print()
                print(f"Now printing content of file '{filename}'")
            with open(r"C:\Users\harih\OneDrive\Documents\1st Semester\ICS-Lab Tasks\task03\Contents_in_zip.txt",'wb') as f:
                for line in archive.read(filename).split(b"\n"):
                
                    f.write(line)
                    print(line)
            return True
    except BadZipfile:
        return False
    except ValueError:
        return False





def main():

    with open(r"C:\Users\harih\OneDrive\Documents\1st Semester\ICS-Lab Tasks\task03\XOR.zip.crypt", "rb") as file:
        cyphertext_bytes = file.read() 
    # Known header values for an encrypted ZIP file
    signature = b'\x50\x4B\x03\x04'      # ZIP Signature - 4 bytes
    version = b'\x0A\x00'                # ZIP Version - 2 bytes 
    flags = b'\x00\x00\x00\x00'                  # ZIP Flags - 2 bytes 

    #Taking corresponding bytes from ciphertext to xor with the header values
    cyphertext_bytes_PK = cyphertext_bytes[0:4]
    cyphertext_bytes_mutate = cyphertext_bytes[4:6]
    cyphertext_rest_of_bytes = cyphertext_bytes[6:10]
    cyphertet_extra_bytes = cyphertext_bytes[10:12]
    known_header = signature + version + flags
    
   
    for i in range(65535):
        """ Brute forcing last two bytes"""
        version_bytes_to_mutate = i.to_bytes(2,byteorder='big')
        
        resulting_bytes = xor_bytes(signature, cyphertext_bytes_PK) + \
                  xor_bytes(version, cyphertext_bytes_mutate) + \
                  xor_bytes(flags, cyphertext_rest_of_bytes) + \
            xor_bytes(version_bytes_to_mutate, cyphertet_extra_bytes)
        decrypted_bytes = try_to_decrypt(cyphertext_bytes,resulting_bytes)
        #Trying to open the zip file with the decrypted bytes
        if is_zipfile(input_bytes=decrypted_bytes, key_to_try =resulting_bytes):
                #if the file is opened the key is stored
                with open(r"C:\Users\harih\OneDrive\Documents\1st Semester\ICS-Lab Tasks\task03\XOR.key", "wb") as key_found:

                    hex_key = resulting_bytes.hex().encode("utf-8")
                    key_found.write(hex_key)
                break
        




if __name__ == "__main__":
    # executes only if run as a script
    main()