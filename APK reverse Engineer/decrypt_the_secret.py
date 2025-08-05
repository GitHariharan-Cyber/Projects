from Crypto.Cipher import AES
import base64
import binascii

# Hex string key from the app
key_hex = "8d127684cbc37c17616d806cf50473cc"
key = binascii.unhexlify(key_hex)

# Base64 encoded ciphertext from the app
ciphertext_b64 = "5UJiFctbmgbDoLXmpL12mkno8HT4Lv8dlat8FxR2GOc="
ciphertext = base64.b64decode(ciphertext_b64)

# Set up AES decryption
cipher = AES.new(key, AES.MODE_ECB)
decrypted = cipher.decrypt(ciphertext)

# Remove padding (PKCS7)
pad_len = decrypted[-1]
plaintext = decrypted[:-pad_len]

print("Secret is:", plaintext.decode())
