#!/usr/bin/env python3
"""
Cloud watchdog runner for GitHub Actions.
Checks Kaggle kernel status and relaunches if stopped.
"""

import os
import subprocess
import sys
import time

KERNEL_SLUG = "turborx/project-chronos-ppo-self-play-training"
KAGGLE_TOKEN = os.environ.get("KAGGLE_API_TOKEN")


def get_status() -> str:
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "status", KERNEL_SLUG],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        return res.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def relaunch_kernel() -> bool:
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "push", "-p", "training"],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        print(f"[Recovery] kaggle kernels push output: {res.stdout.strip()} {res.stderr.strip()}")
        return res.returncode == 0
    except Exception as e:
        print(f"[Recovery] Error relaunching: {e}")
        return False


def main():
    status_output = get_status()
    print(f"Current Kaggle Status: {status_output}")

    if "KernelWorkerStatus.RUNNING" in status_output or "status \"running\"" in status_output.lower():
        print("[✓] Kernel is actively training on GPU. No intervention needed.")
        sys.exit(0)

    print("[!] Kernel is not running. Initiating cloud recovery sequence...")
    success = relaunch_kernel()
    if success:
        print("[✓] Kernel successfully relaunched.")
        sys.exit(0)
    else:
        print("[✗] Failed to relaunch kernel.")
        sys.exit(1)


if __name__ == "__main__":
    main()
