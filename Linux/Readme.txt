# Linux Namespace + chroot Sandbox (Educational Project)

This repository demonstrates how to build a lightweight, educational Linux sandbox using:

- PID namespaces  
- Mount namespaces  
- UTS namespaces (isolated hostname)  
- `chroot`  
- privilege dropping (uid/gid nobody)  
- resource limits (CPU, memory, open files)  
- minimal root filesystem construction  
- BusyBox or debootstrap environments  

This is NOT a production security sandbox — but an excellent learning tool for:
- OS security concepts  
- containers / container runtimes  
- process isolation  
- Linux namespaces  
- secure software design principles  

---

##  Repository Structure

linux-sandbox/
├─ .gitignore
├─ LICENSE
├─ README.md
├─ py_sandbox.py
├─ populate_rootfs.sh
├─ setup_sandbox.sh
├─ tests/
│ └─ test_sanity.sh
└─ SECURITY.md




---

##  Quick Start (Ubuntu / Debian)

### 1. Clone the repo

```bash
git clone https://github.com/GitHariharan-Cyber/Projects/blob/main/Linux
cd linux-sandbox


2. Build a minimal BusyBox root filesystem
sudo ./setup_sandbox.sh /srv/sandbox-root busybox

3. Run the sandbox
sudo ./py_sandbox.py /srv/sandbox-root /bin/sh -c "id; hostname; ps aux"


Expected output:

uid=65534 gid=65534
sandboxed
PID   USER     COMMAND

1 nobody   {ps} ...

