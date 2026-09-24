#!/usr/bin/env python3
"""
Autonomous training watchdog.
Monitors Kaggle training job, downloads checkpoints on completion,
syncs to Hugging Face, and relaunches the training job.
"""

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
WATCHDOG_LOG = LOG_DIR / "watchdog.log"
TRAINING_DIR = BASE_DIR / "training"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

KAGGLE_KERNEL_SLUG = "turborx/project-chronos-ppo-self-play-training"
HF_REPO_ID = "TurboRx/chronos-randbats"
HF_TOKEN = os.environ.get("HF_TOKEN")
KAGGLE_TOKEN = os.environ.get("KAGGLE_API_TOKEN")
POLL_INTERVAL_SEC = 300


def log(msg: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(WATCHDOG_LOG, "a") as f:
        f.write(line + "\n")


def get_kernel_status() -> str:
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "status", KAGGLE_KERNEL_SLUG],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        for line in res.stdout.strip().split("\n"):
            if "status" in line.lower():
                return line.strip()
        return res.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def download_latest_checkpoint() -> bool:
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN
    out_dir = CHECKPOINTS_DIR / "kaggle_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        res = subprocess.run(
            ["kaggle", "kernels", "output", KAGGLE_KERNEL_SLUG, "-p", str(out_dir)],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        return res.returncode == 0
    except Exception as e:
        log(f"Error downloading output: {e}")
        return False


def backup_to_huggingface() -> bool:
    ckpt = CHECKPOINTS_DIR / "kaggle_output" / "checkpoint_latest.pkl"
    if not ckpt.exists():
        ckpt = CHECKPOINTS_DIR / "bc_checkpoint_latest.pkl"

    if not ckpt.exists():
        return False

    try:
        cmd = [
            sys.executable,
            str(BASE_DIR / "scripts" / "deploy_hf.py"),
            "--repo-id",
            HF_REPO_ID,
            "--checkpoint",
            str(ckpt),
            "--token",
            HF_TOKEN,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return res.returncode == 0
    except Exception as e:
        log(f"Error backing up to Hugging Face: {e}")
        return False


def relaunch_kaggle_kernel() -> bool:
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN

    latest_ckpt = CHECKPOINTS_DIR / "kaggle_output" / "checkpoint_latest.pkl"
    if latest_ckpt.exists():
        target_ckpt_dir = TRAINING_DIR / "checkpoints"
        target_ckpt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest_ckpt, target_ckpt_dir / "checkpoint_latest.pkl")

    try:
        res = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(TRAINING_DIR)],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        return res.returncode == 0
    except Exception as e:
        log(f"Error relaunching kernel: {e}")
        return False


def run_watchdog():
    log(f"Watchdog started for {KAGGLE_KERNEL_SLUG} (poll interval: {POLL_INTERVAL_SEC}s)")

    while True:
        try:
            status_line = get_kernel_status()
            log(f"Status: {status_line}")

            status_lower = status_line.lower()
            if "complete" in status_lower or "stopped" in status_lower or "error" in status_lower:
                log("Kernel stopped. Executing recovery: download -> backup -> relaunch")
                download_latest_checkpoint()
                backup_to_huggingface()
                relaunch_kaggle_kernel()
                time.sleep(600)
                continue

        except Exception as e:
            log(f"Watchdog loop error: {e}")

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    run_watchdog()
