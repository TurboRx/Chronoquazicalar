#!/usr/bin/env python3
"""colab_auto_supervisor.py

24/7 Autonomous Supervisor for Chronos Colab Training on Azure VM.
Monitors the self-play training process. If Colab hits its 12-hour limit,
disconnects, or terminates, the supervisor automatically requests a new T4 GPU,
re-uploads the training scripts, and resumes training from the latest Hugging Face checkpoint.
"""

import datetime
import os
import subprocess
import time

LOG_FILE = os.path.expanduser("~/colab_supervisor.log")
TRAIN_LOG = os.path.expanduser("~/colab_train.log")
SELFPLAY_SCRIPT = os.path.expanduser("~/selfplay_ppo.py")
TRANSFORMER_SCRIPT = os.path.expanduser("~/transformer_policy.py")
START_SCRIPT = os.path.expanduser("~/start_training.py")


def log(msg):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now} UTC] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def is_tmux_session_alive(session_name="colab-train"):
    res = subprocess.run(["tmux", "has-session", "-t", session_name], capture_output=True)
    return res.returncode == 0


def get_train_log_age_seconds():
    if not os.path.exists(TRAIN_LOG):
        return 999999
    try:
        mtime = os.path.getmtime(TRAIN_LOG)
        return time.time() - mtime
    except Exception:
        return 999999


def get_latest_train_log_line():
    if not os.path.exists(TRAIN_LOG):
        return ""
    try:
        res = subprocess.run(["tail", "-n", "1", TRAIN_LOG], capture_output=True, text=True)
        return res.stdout.strip()
    except Exception:
        return ""


def restart_colab_training():
    log("[Supervisor] === Initiating Colab Self-Healing & Auto-Restart Procedure ===")

    # 1. Kill stale tmux session if present
    log("[Supervisor] Cleaning up any stale tmux session...")
    subprocess.run(["tmux", "kill-session", "-t", "colab-train"], capture_output=True)

    # 2. Stop stale colab session
    log("[Supervisor] Stopping previous Colab session 'chronos-train'...")
    subprocess.run(["colab", "stop", "-s", "chronos-train"], capture_output=True)
    time.sleep(5)

    # 3. Acquire new Colab GPU session (loop with backoff if quota/rate-limited)
    attempt = 0
    while True:
        attempt += 1
        log(f"[Supervisor] Requesting new Colab GPU runtime (T4) [Attempt #{attempt}]...")
        try:
            res = subprocess.run(
                ["colab", "new", "-s", "chronos-train", "--gpu", "T4"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            output = (res.stdout + " " + res.stderr).strip()
            if res.returncode == 0 and ("READY" in output or "chronos-train" in output):
                log(f"[Supervisor] Successfully acquired Colab GPU session: {output}")
                break
            else:
                log(f"[Supervisor] Colab allocation returned: {output}. Retrying in 180 seconds...")
                time.sleep(180)
        except subprocess.TimeoutExpired:
            log("[Supervisor] Request timed out after 300s. Retrying in 60 seconds...")
            time.sleep(60)
        except Exception as e:
            log(f"[Supervisor] Unexpected error during 'colab new': {e}. Retrying in 60s...")
            time.sleep(60)

    # 4. Upload fresh training scripts
    log("[Supervisor] Uploading training code to Colab /content...")
    upload_attempts = 0
    while upload_attempts < 5:
        upload_attempts += 1
        try:
            r1 = subprocess.run(
                ["colab", "upload", "-s", "chronos-train", SELFPLAY_SCRIPT, "/content/selfplay_ppo.py"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            r2 = subprocess.run(
                ["colab", "upload", "-s", "chronos-train", TRANSFORMER_SCRIPT, "/content/transformer_policy.py"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if r1.returncode == 0 and r2.returncode == 0:
                log("[Supervisor] Training scripts uploaded successfully.")
                break
            else:
                log(
                    f"[Supervisor] Upload attempt #{upload_attempts} failed: {r1.stderr} {r2.stderr}. Retrying in 10s..."
                )
                time.sleep(10)
        except Exception as e:
            log(f"[Supervisor] Error during upload: {e}. Retrying in 10s...")
            time.sleep(10)

    # 5. Start training inside tmux
    log("[Supervisor] Launching 'start_training.py' inside detached tmux session 'colab-train'...")
    subprocess.run(["tmux", "new-session", "-d", "-s", "colab-train", f"python3 {START_SCRIPT}"], check=True)
    log("[Supervisor] Training launched! Monitoring will continue...")
    time.sleep(60)


def main():
    log("==============================================================")
    log("[Supervisor] Chronos 24/7 Training Supervisor Started on Azure VM")
    log("==============================================================")

    last_heartbeat = 0

    while True:
        try:
            tmux_alive = is_tmux_session_alive("colab-train")
            log_age = get_train_log_age_seconds()
            now = time.time()

            # If tmux is alive AND log is fresh (< 300 seconds old = actively generating updates)
            if tmux_alive and log_age < 300:
                if now - last_heartbeat > 300:
                    last_heartbeat = now
                    latest_line = get_latest_train_log_line()
                    log(f"[Heartbeat] Training ACTIVE & HEALTHY. Log age: {log_age:.1f}s. Latest: {latest_line}")
            else:
                # Training has stopped or stalled!
                reason = []
                if not tmux_alive:
                    reason.append("tmux session 'colab-train' exited")
                if log_age >= 300:
                    reason.append(f"training log has not updated in {log_age:.1f}s")
                log(f"[ALERT] Training issue detected: {', '.join(reason)}.")
                restart_colab_training()
                last_heartbeat = time.time()

        except Exception as e:
            log(f"[Supervisor] Unexpected exception in main loop: {e}")

        time.sleep(30)


if __name__ == "__main__":
    main()
