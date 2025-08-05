import random
from collections import Counter
from ngram_score import ngram_score


# Load the English 4-gram scoring model
scorer = ngram_score(r"C:\Users\harih\OneDrive\Documents\1st Semester\ICS-Lab Tasks\task01\english_quadgrams.txt")

# Frequency of letters in English (for reference in initializing the key)
english_freq_order =  "ETAOINSHRDLCUMWFGYPBVKJXQZ"


def key_formulation(ciphertext):
    """ Initialize a key based on frequency analysis of the ciphertext. """
    frequency_in_cipher = Counter([char for char in ciphertext if char.isalpha()]) #taking only the alphabets
    sorted_cipher = [char for char,i in frequency_in_cipher.most_common()]  #Sorting the alphabets based on the frequency
    
    initial_key={}
    orderedkey =''
    # Create initial key mapping from frequency analysis
    for i,char in enumerate(sorted_cipher):
        if i < len(english_freq_order):
            initial_key[char]=english_freq_order[i]   
    
    # Fill remaining letters randomly for unmapped characters
    remaining_keys = set(english_freq_order)-set(initial_key.values())
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if char not in initial_key:
            initial_key[char]=remaining_keys.pop()
            orderedkey+=initial_key.get(char,char)
        else:
            orderedkey+=initial_key.get(char,char)
    #The key is ordered in alphabetic order"
    return orderedkey


def swap_chars(key):
    """ Creates a new key by swapping two random characters in the current key. """
    key_list = list(key)
    i, j = random.sample(range(26),2)
    key_list[i], key_list[j] = key_list[j], key_list[i]
    return ''.join(key_list)
    
            
    
def hillClimb(ciphertext):
    """ Hill-climbing algorithm to find the key that decrypts the ciphertext. """
    current_key = key_formulation(ciphertext)
    #Table creeated to map the alphabets with key obtained
    lookup_table =str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ",current_key)
    current_decrypted = ciphertext.translate(lookup_table)
    current_score = scorer.score(current_decrypted)

    for iteration in range(10000):
        """The iteration value is just updated by trial and error until I got the meaningful plaintext"""
        updated_key = swap_chars(current_key)
        lookup_table = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ",updated_key)
        updated_decrypted = ciphertext.translate(lookup_table)
        updated_score = scorer.score(updated_decrypted)

        #Checking if the old score is lesser than the new score and then updating score for further iterations
        if updated_score > current_score:
            current_key = updated_key
            current_score = updated_score
            current_decrypted = updated_decrypted
    return current_decrypted, current_key,updated_score

with open(r"C:\Users\harih\OneDrive\Documents\1st Semester\ICS-Lab Tasks\task01\Subst.txt","r") as f:
    ciphertext= f.read().upper() #Read the ciphertext and convert it into upper case

# Run the hill-climbing decryption
plaintext , key ,score = hillClimb(ciphertext)

# Save the final results
with open(r"C:\Users\harih\OneDrive\Documents\1st Semester\ICS-Lab Tasks\task01\Plain.txt","w") as f:
    f.write(plaintext)

with open(r"C:\Users\harih\OneDrive\Documents\1st Semester\ICS-Lab Tasks\task01\subst.key","w") as f:
    hex_key = key.encode("utf-8").hex()
    f.write(hex_key)
