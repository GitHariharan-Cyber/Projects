
Main Task:

Step 1: Connect to the BTU -Vpn if not connected in eduroam network

Step 2: Login to the server using ssh

Step 3: Open two terminals 

Step 4: Verify the connection using sudo tcpdump -i eth1 # To start the packet capture on the ethernet 1 interface  

Step 5: using the command sudo raw_packet eth1 targetMAC 0x0806 payload.bin

#raw_packet will be sent as the ARP reply to Alice

Step 6: Capture the traffic in capture.pcap file to analyse the replies from Alice 



ARP Packet is crafted in the following sequence:


# build payloads
# source: https://en.wikipedia.org/wiki/Address_Resolution_Protocol

#  Hardware type (HTYPE): This field specifies the network link protocol type. Example: Ethernet is 1.Represented in hex as 0x0001

# Protocol type (PTYPE): This field specifies the internetwork protocol for which the ARP request is intended. For IPv4, this has the value 0x0800.

# Hardware length (HLEN): Length (in octets) of a hardware address. Ethernet address length is 0x06.

# Protocol length (PLEN): Length (in octets) of internetwork addresses. The internetwork protocol is specified in PTYPE. Example: IPv4 address length is 0x04.

# Operation: Specifies the operation that the sender is performing: 1 for request, 2 for reply.

# Sender hardware address (SHA): Media address of the sender. In an ARP request this field is used to indicate the address of the host sending the request. In an ARP reply this field is used to indicate the address of the host that the request was looking for.

# Sender protocol address (SPA): Internetwork address of the sender.
# Sender protocol address (SPA): Internetwork address of the sender.
# Target protocol address (TPA): Internetwork address of the intended receiver.

So for alice the payload will be :

Ethernet : 0x0001
Protocol type : 0x0800
Hardware size :6
Protocol size:4
Opcode:2
Sender MAC address: 02:42:0a:0a:1b:04
Sender IP address: 0a:0a:1b:03
Target MAC address: 02:42:0a:0a:1b:02
Target IP address: 0a:0a:1b:02

The payload is constructed and converted into binary payload.py file in my local system.


Step 7: Send this file to the server using scp command 

Step 8: After sending this packet as reply  using the mitm_script.sh file ,we have to store the output reply from Alice in file using command 

sudo tcpdump -i eht1 -w capture.pcap 



Step 9:Search for the CTF flag using grep command in the captured file using the command 

       strings capture.pcap | grep "CTF"

Step 10: Notedown the CTF{secret-azCKQOJ3NTDzgeyJMMnY}


 
Step 11: Forward the reply to bob. As the ip forwarding is already configured ,ensure the packets reach to bob.


Similarly follow the same steps for listening to bob's message by slightly modigfying the payload file and forward it to alice.









 