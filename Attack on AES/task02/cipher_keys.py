# cipher_keys.py

def read_file(filename):
    """Reads the encypted file for input"""
    with open(filename, "rb") as f:
        return f.read()

def find_keys(ciphertext, plaintext1, plaintext2):
    
    k1 = bytearray()
    k2 = bytearray()
    
    # Ensure all files have the same length for XOR operations
    assert len(ciphertext) == len(plaintext1) == len(plaintext2), \
        """Ciphertext and plaintext files must be of the same length."""
    
    for i in range(len(ciphertext)):
        """ Xor performed between each bit of cipher and plain text"""
        k1.append(ciphertext[i] ^ plaintext1[i])
        k2.append(ciphertext[i] ^ plaintext2[i])
    
    return bytes(k1), bytes(k2)

def save_key(filename, key_data):
    """Saves the given key """
    with open(filename, "wb") as f:
        hex_key = key_data.hex().encode("utf-8")
        f.write(hex_key)

def decrypt(ciphertext, key):
    """Decrypts the ciphertext using the given key by performing XOR operation."""
    return bytes(c ^ k for c, k in zip(ciphertext, key))

def verify_decryption(ciphertext, key1, key2, plaintext1, plaintext2):
    """Verifies if the keys decrypt the ciphertext to the correct plaintexts."""
    decrypted1 = decrypt(ciphertext, key1)
    decrypted2 = decrypt(ciphertext, key2)
    
    return decrypted1 == plaintext1, decrypted2 == plaintext2

def main():
    # Read the files
    ciphertext = read_file(r"C:\Users\harih\OneDrive\Documents\1st Semester\ICS-Lab Tasks\task02\cipher.crypt")
    plaintext1 = read_file(r"C:\Users\harih\OneDrive\Documents\1st Semester\ICS-Lab Tasks\task02\plaintext1.txt")
    plaintext2 = read_file(r"C:\Users\harih\OneDrive\Documents\1st Semester\ICS-Lab Tasks\task02\plaintext2.txt")


    # Generate the keys
    k1, k2 = find_keys(ciphertext, plaintext1, plaintext2)
    # Save the keys to files
    save_key(r"C:\Users\harih\OneDrive\Documents\1st Semester\ICS-Lab Tasks\task02\k1.key", k1)
    save_key(r"C:\Users\harih\OneDrive\Documents\1st Semester\ICS-Lab Tasks\task02\k2.key", k2)

    print("Keys k1 and k2 generated and saved to k1.key and k2.key.")

    # Verify the decryption
    is_valid1, is_valid2 = verify_decryption(ciphertext, k1, k2, plaintext1, plaintext2)
    
    if is_valid1 and is_valid2:
        print("Verification successful: Both keys decrypt correctly.")
    else:
        print("Verification failed: One or both keys do not decrypt correctly.")

if __name__ == "__main__":
    main()
