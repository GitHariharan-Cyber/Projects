#!/bin/bash

echo "===== SELinux Secure Service Demo ====="
echo

echo "[0] Starting vehicle_log service"
systemctl start vehicle_log
sleep 1
echo

echo "[1] Checking SELinux mode"
getenforce
echo

echo "[2] Checking process domain"
ps -eZ | grep vehicle_log || echo "Process exited (expected for one-shot)"
echo

echo "[3] Checking executable label"
ls -Z /usr/local/bin/vehicle_log
echo

echo "[4] Checking data directory label"
ls -Zd /var/lib/vehicle_log
echo

echo "[5] Checking written data file"
ls -Z /var/lib/vehicle_log/data.txt 2>/dev/null || echo "data file not present"
echo

echo "[6] Recent AVC denials (proof of enforcement)"
ausearch -m AVC -ts recent | tail -5
echo

echo "===== Demo Complete ====="
