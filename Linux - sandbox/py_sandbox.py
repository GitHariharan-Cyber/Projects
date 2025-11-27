
---

# ✅ **4. `py_sandbox.py`**

Create: **`linux-sandbox/py_sandbox.py`**

```python
#!/usr/bin/env python3
"""
py_sandbox.py - Educational Linux sandbox using namespaces + chroot.

Features:
 - PID namespace
 - Mount namespace
 - UTS namespace (hostname)
 - chroot isolation
 - privilege drop (nobody)
 - resource limits (CPU, memory, file handles)

This script must run as root.

Usage example:
    sudo ./py_sandbox.py /srv/sandbox-root /bin/sh -c "id; ps"
"""
import os
import sys
import subprocess
import resource

CLONE_NEWNS  = 0x00020000
CLONE_NEWPID = 0x20000000
CLONE_NEWUTS = 0x04000000

def die(msg):
    print("ERROR:", msg, file=sys.stderr)
    sys.exit(1)

def mount_proc(root):
    path = os.path.join(root, "proc")
    os.makedirs(path, exist_ok=True)
    subprocess.check_call(["mount", "-t", "proc", "proc", path])

def unmount_proc(root):
    path = os.path.join(root, "proc")
    subprocess.call(["umount", path])

def drop_privileges():
    import pwd, grp
    try:
        uid = pwd.getpwnam("nobody").pw_uid
        gid = grp.getgrnam("nogroup").gr_gid
    except KeyError:
        uid = 65534
        gid = 65534

    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)

def set_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    mem = 300 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))

def main():
    if os.geteuid() != 0:
        die("Must run as root")

    if len(sys.argv) < 3:
        die("Usage: py_sandbox.py <rootfs> <command...>")

    rootfs = os.path.abspath(sys.argv[1])
    cmd = sys.argv[2:]

    print("[*] Unsharing namespaces...")
    os.unshare(CLONE_NEWNS | CLONE_NEWPID | CLONE_NEWUTS)
    os.sethostname(b"sandboxed")

    pid = os.fork()
    if pid != 0:
        _, status = os.waitpid(pid, 0)
        print("[*] Child exited:", status)
        return

    subprocess.call(["mount", "--make-rprivate", "/"])
    mount_proc(rootfs)

    print("[*] chroot ->", rootfs)
    os.chroot(rootfs)
    os.chdir("/")

    print("[*] Dropping privileges...")
    drop_privileges()

    set_limits()

    print("[*] Exec:", cmd)
    os.execvp(cmd[0], cmd)

if __name__ == "__main__":
    main()
