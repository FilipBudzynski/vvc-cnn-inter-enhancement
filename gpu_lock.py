"""Exclusive-GPU lock shared by all training/evaluation entrypoints."""

import fcntl
import os
import time

LOCK_PATH = "/tmp/vvc_gpu.lock"
_handle = None  # keep the fd alive for the process lifetime


def acquire_gpu(name: str):
    global _handle
    _handle = open(LOCK_PATH, "w")
    t0 = time.time()
    while True:
        try:
            fcntl.flock(_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() - t0 < 5 or int(time.time() - t0) % 600 < 2:
                print(f"[gpu_lock] {name}: waiting for exclusive GPU "
                      f"({(time.time()-t0)/60:.0f} min)", flush=True)
            time.sleep(20)
    _handle.write(f"{os.getpid()} {name}\n")
    _handle.flush()
    print(f"[gpu_lock] {name}: acquired after {(time.time()-t0)/60:.1f} min",
          flush=True)
