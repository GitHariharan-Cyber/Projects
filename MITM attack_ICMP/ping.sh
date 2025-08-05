#!/bin/bash

# check if host-192.168.60.5 can reach other subnet
sudo docker exec -it 80d ping -c1 10.9.0.105

#check if another machine is pingable within the subnet
sudo docker exec -it 80d ping -c1 192.168.60.6
