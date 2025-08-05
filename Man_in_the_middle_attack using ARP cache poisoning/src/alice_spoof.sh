#!/bin/bash

while true;
do
        sudo /usr/bin/raw_packet eth1 02:42:0a:0a:1b:02 0x0806 payload_alice_new.bin
        sleep 1
done
