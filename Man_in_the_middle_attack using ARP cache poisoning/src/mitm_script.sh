#!/bin/bash
while true;
do
        sudo /usr/bin/raw_packet eth1 02:42:0a:0a:1b:02 0x0806 payload.bin
        sudo /usr/bin/raw_packet eth1 02:42:0a:0a:1b:03 0x0806 arp_payload_bob.bin
        sleep 6
done
