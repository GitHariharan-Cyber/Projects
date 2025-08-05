from Crypto.Cipher import AES
from collections import Counter
import math

#calculate the shannon_entropy for the decrypted text 
def shanon_entropy(text):
    length = len(text)
    if length == 0:
        return 0
    Noofval = Counter(text)     #counts the occurance of each unique characters
    entropy = 0
    for count in Noofval.values():
        probability = count / length
        entropy -= probability * math.log2(probability)   #calculating entropy value
    return entropy

# class to perform AES decryption 
def aes_decryption_with_key(ciphertxt, InitVector, keys):
    cipher = AES.new(keys, AES.MODE_CBC, InitVector)
    return cipher.decrypt(ciphertxt)
    
# Brute force AES-CBC with weak key first 16 bits chosen
def weakkey_bruteforce_aes(InitVector, ciphertxt):
    max_value = 7.0
    Validtxts = []
    for i in range(2**16):
        keys = i.to_bytes(2, byteorder='big') + b'\x00' * 14         #packing the key with first 2bytes of data and the rest 14 bytes with 0
        Decryptedtxt = aes_decryption_with_key(ciphertxt, InitVector, keys)
        entropy_val = shanon_entropy(Decryptedtxt)                   #calling class shannon_entropy to calculate the entropy

        if entropy_val < max_value:
            max_value = entropy_val                                 #comparing with all the entropy values and choosing the least one
            Validtxts.append((Decryptedtxt, keys, entropy_val))

    # Find the best candidate based on entropy
    Meaningful_txt =min(Validtxts)
    print(Meaningful_txt)
    return Meaningful_txt

# Read the encrypted file from the local directory
with open(r"C:\Users\harih\OneDrive\Documents\1st Semester\ICS-Lab Tasks\task01\Subst-Rijndael.crypt", "rb") as f:
    InitVector = f.read(16)    #read the first 16 bytes for the Initialization vector used in CBC
    ciphertxt = f.read()       #storing the rest of the words which is the cipher text

# Run the brute-force AES decryption 
result = weakkey_bruteforce_aes(InitVector, ciphertxt)

#checking if the result is not empty
if result is not None: 
    plaintext, key, entropy_val = result  # Unpack the three values from the result which contains plaintext,key and corresponding entropy value
    with open(r"C:\Users\harih\OneDrive\Documents\1st Semester\ICS-Lab Tasks\task01\aes.key", "w") as f:
        f.write(key.hex())      #key is written into aes.key file
        print("AES key written to aes.key")
        
    with open(r"C:\Users\harih\OneDrive\Documents\1st Semester\ICS-Lab Tasks\task01\Subst.txt", "wb") as f:
        f.write(plaintext)        #plaintext is written in the subst.txt
        print("Decrypted text written to Subst.txt")
else:
    print("No valid plaintext found.")

