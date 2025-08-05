#!/usr/bin/env python3
from scapy.all import *
import argparse

# Creates an ArgumentParser object to handle command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--sourceaddress", "-src", help="Source IP", required=True)   # Adds required argument: source IP address to filter traffic from
parser.add_argument("--pattern", "-f", help="Pattern to find", required=False)  # pattern in the packet's payload that should be replaced
parser.add_argument("--stringreplacement", "-r", help="Replacement string", required=False)  #string that will replace the pattern if found
args = parser.parse_args()

source_ip = args.sourceaddress
pattern = args.pattern
replacement = args.stringreplacement

def packet_callback(pkt):
    # Only proceed if the packet contains IP, UDP, and Raw layers (i.e., payload data)
    if pkt.haslayer(IP) and pkt.haslayer(UDP) and pkt.haslayer(Raw):
        raw = pkt[Raw].load
        new_payload = raw   # extract original payload

        # Check if pattern and replacement are provided and pattern exists in payload
        if pattern and replacement and pattern.encode() in raw:
            new_payload = raw.replace(pattern.encode(), replacement.encode())  #Find the pattern and replace,since it is udp no need to worry about the length
            print(f"[+] Replaced '{pattern}' → '{replacement}'")
            # Clone the original IP packet (as bytes) into a new Scapy IP object
            new_pkt = IP(src=pkt[IP].src, dst=pkt[IP].dst) / \
                      UDP(sport=pkt[UDP].sport, dport=pkt[UDP].dport) / \
                      new_payload
            # Send the new packet with modified payload; 'verbose=False' suppresses extra output
            send(new_pkt, verbose=False)

# Start sniffing on interface eth0 using the defined BPF filter
# For each captured packet, call packet_callback(pkt)
sniff(iface="eth0", filter=f"udp and src host {source_ip}", prn=packet_callback)