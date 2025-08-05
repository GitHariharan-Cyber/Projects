#!/bin/bash
while true;
do
        sudo /usr/bin/raw_packet eth1 02:42:0a:0a:1b:03 0x0806 payload_bob_new.bin
        sleep 1
done
