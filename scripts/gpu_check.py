"""Is anyone else computing on the GPU? Exit 0 = free, 1 = busy, 2 = no CUDA.

Primary source: `nvidia-smi --query-compute-apps` (compute processes only, so
the desktop's Xorg / gnome-shell do not count). If nvidia-smi fails (it did on
2026-11-15 with "Driver/library version mismatch" until a reboot), fall back
to listing processes that hold /dev/nvidia<N>, ignoring display processes,
plus free memory reported by torch.
"""

from __future__ import annotations

import os
import subprocess
import sys

DISPLAY_PROCS = ("Xorg", "Xwayland", "gnome-shell", "gnome-remote-desktop", "xrdp", "kwin", "plasmashell")


def _smi(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True)


def compute_apps_smi() -> list[str] | None:
    r = _smi(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    if r.returncode != 0:
        return None
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def gpu_holders_proc() -> list[str]:
    me, out = os.getpid(), []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        if int(pid) == me:
            continue
        try:
            targets = [os.readlink(f"/proc/{pid}/fd/{fd}") for fd in os.listdir(f"/proc/{pid}/fd")]
        except OSError:
            continue
        if any(t.startswith("/dev/nvidia") and t[11:].isdigit() for t in targets):
            cmd = subprocess.run(["ps", "-o", "user=,etime=,args=", "-p", pid],
                                 capture_output=True, text=True).stdout.strip()
            if not any(d in cmd for d in DISPLAY_PROCS):
                out.append(f"{pid} {cmd[:150]}")
    return out


def main() -> int:
    status = _smi(["--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"])
    apps = compute_apps_smi()
    if apps is not None:
        print("nvidia-smi:", status.stdout.strip())
        print(f"compute processes on the GPU: {len(apps)}")
        for a in apps:
            print("  ", a)
        busy = bool(apps)
    else:
        err = (status.stderr or status.stdout).strip().splitlines()
        print("nvidia-smi unavailable:", err[0] if err else "?", "-> fallback to /proc + torch")
        holders = gpu_holders_proc()
        print(f"non-display processes holding the GPU: {len(holders)}")
        for h in holders:
            print("  ", h)
        busy = bool(holders)

    import warnings

    warnings.filterwarnings("ignore", message="Can't initialize NVML")
    import torch

    if not torch.cuda.is_available():
        print("torch: CUDA not available")
        return 2
    free, total = torch.cuda.mem_get_info()
    print(f"torch: {torch.cuda.get_device_name(0)}  free {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
    busy = busy or free < 0.85 * total
    print("GPU looks", "BUSY" if busy else "free")
    return 1 if busy else 0


if __name__ == "__main__":
    sys.exit(main())
