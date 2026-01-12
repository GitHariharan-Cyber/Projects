# SELinux Secure System Service (Automotive / Embedded Demo)

## Overview
This project demonstrates how to confine a Linux system service using SELinux.
The service runs in its own SELinux domain, is allowed to write only to a
dedicated data directory, and is blocked from accessing sensitive system files
such as `/etc/shadow`.

This design mirrors security models used in automotive infotainment systems,
Android Automotive, and Yocto-based embedded Linux platforms.

---

## Key Security Concepts Demonstrated
- SELinux domain transitions
- systemd trusted execution
- Least-privilege filesystem access
- Mandatory Access Control (MAC)
- AVC denial analysis

---

## Project Components

### Application
- `vehicle_log.c`  
  Writes data to `/var/lib/vehicle_log` and attempts to read `/etc/shadow`.

### systemd Service
- Runs the application as a managed system service.
- Triggers SELinux domain transition.

### SELinux Policy
- `vehicle_log_t` – confined process domain
- `vehicle_log_exec_t` – executable label
- `vehicle_log_data_t` – restricted data directory

### Demo Script
- Validates SELinux enforcement in under 5 minutes.

---

## Build & Install

```bash
gcc app/vehicle_log.c -o vehicle_log
sudo cp vehicle_log /usr/local/bin/vehicle_log
sudo chmod 755 /usr/local/bin/vehicle_log
