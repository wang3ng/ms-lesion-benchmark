"""Who is using the GPU? Works even when nvidia-smi fails.

On the lab machine nvidia-smi can fail with "Driver/library version
mismatch" after a driver upgrade without reboot, while CUDA still works.
This reports free memory via torch and lists processes holding /dev/nvidia*.
Exit code 0 = GPU looks free, 1 = someone else is on it, 2 = no CUDA.
"""

from __future__ import annotations

import os
import subprocess
import sys


def gpu_holders() -> list[tuple[int, str]]:
    me = os.getpid()
    out = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        if int(pid) == me:
            continue
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if target.startswith("/dev/nvidia") and target[len("/dev/nvidia"):].isdigit():
                cmd = subprocess.run(["ps", "-o", "user=,etime=,args=", "-p", pid],
                                     capture_output=True, text=True).stdout.strip()
                out.append((int(pid), cmd[:150]))
                break
    return out


def main() -> int:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True)
    print("nvidia-smi:", smi.stdout.strip() or smi.stderr.strip().splitlines()[0])

    holders = gpu_holders()
    print(f"processes holding the GPU (before torch init): {len(holders)}")
    for pid, cmd in holders:
        print(f"  {pid}  {cmd}")

    import torch

    if not torch.cuda.is_available():
        print("torch: CUDA not available")
        return 2
    free, total = torch.cuda.mem_get_info()
    print(f"torch: {torch.cuda.get_device_name(0)}  free {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
    busy = bool(holders) or free < 0.9 * total
    print("GPU looks", "BUSY" if busy else "free")
    return 1 if busy else 0


if __name__ == "__main__":
    sys.exit(main())
