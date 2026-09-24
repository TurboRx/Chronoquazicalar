#!/usr/bin/env python3
"""
scripts/check_progress.py
Displays training progress, games played, win rate, and Elo progression.
"""

import json
import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_LOCAL = BASE_DIR / "checkpoints" / "metrics.json"
METRICS_KAGGLE = BASE_DIR / "checkpoints" / "kaggle_output" / "metrics.json"
KAGGLE_KERNEL_SLUG = "turborx/project-chronos-ppo-self-play-training"
KAGGLE_TOKEN = os.environ.get("KAGGLE_API_TOKEN")


def get_remote_status():
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "status", KAGGLE_KERNEL_SLUG],
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
        )
        return res.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def display_progress():
    print("================================================================")
    print("Project Chronos: Training Progress & Elo Monitor")
    print("================================================================")

    # 1. Check Kaggle Kernel Status
    status = get_remote_status()
    print(f"Kaggle Cluster Status: {status}")

    # 2. Check Metrics File
    metrics_file = None
    if METRICS_KAGGLE.exists():
        metrics_file = METRICS_KAGGLE
    elif METRICS_LOCAL.exists():
        metrics_file = METRICS_LOCAL

    if metrics_file and metrics_file.exists():
        try:
            with open(metrics_file, "r") as f:
                data = json.load(f)

            print("\n--- Training Metrics ---")
            print(f"Completed Updates:        {data.get('total_updates', 0):,}")
            print(f"Total Turns Simulated:    {data.get('total_turns', 0):,}")
            print(f"Total Battles Completed:  {data.get('total_battles', 0):,}")
            print(f"Win Rate vs Baseline:     {data.get('win_rate_vs_baseline', 0.0) * 100:.1f}%")
            print(f"Estimated Elo Rating:     {data.get('estimated_elo', 1000):.0f} (Base: 1000)")
            print(f"Policy Loss:              {data.get('policy_loss', 0.0):.4f}")
            print(f"Value Loss:               {data.get('value_loss', 0.0):.4f}")
            print(f"Training Complete:        {'YES' if data.get('is_complete') else 'NO (In Progress)'}")
        except Exception as e:
            print(f"[!] Could not parse metrics file: {e}")
    else:
        print("\n[i] Metrics file will appear as soon as the first 30-minute checkpoint completes.")

    print("================================================================")


if __name__ == "__main__":
    display_progress()
