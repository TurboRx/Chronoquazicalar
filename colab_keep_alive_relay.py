#!/usr/bin/env python3
import datetime
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

CLIENT_ID = os.environ.get("COLAB_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("COLAB_CLIENT_SECRET", "")
REFRESH_TOKEN = os.environ.get("COLAB_REFRESH_TOKEN", "")
ENDPOINT = os.environ.get("COLAB_ENDPOINT", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_METRICS_URL = os.environ.get(
    "HF_METRICS_URL", "https://huggingface.co/TurboRx/chronos-randbats/raw/main/metrics.json"
)
LOG_FILE = os.path.expanduser("~/colab_relay.log")

access_token = None
token_expiry = 0


def log(msg):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now} UTC] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def get_access_token():
    global access_token, token_expiry
    now = time.time()
    if access_token and now < token_expiry - 300:
        return access_token

    log("Refreshing Google OAuth access token...")
    data = urllib.parse.urlencode(
        {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "refresh_token": REFRESH_TOKEN,
            "grant_type": "refresh_token",
        }
    ).encode("utf-8")

    req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode())
        access_token = res["access_token"]
        expires_in = res.get("expires_in", 3600)
        token_expiry = now + expires_in
        log(f"Token refreshed successfully. Valid for {expires_in}s.")
        return access_token


def ping_keep_alive():
    tok = get_access_token()
    url = f"https://colab.research.google.com/tun/m/{ENDPOINT}/keep-alive/?authuser=0"
    headers = {
        "X-Colab-Tunnel": "Google",
        "X-Colab-Client-Agent": "colab-cli",
        "Authorization": f"Bearer {tok}",
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return True, f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP Error {e.code}: {e.reason}"
    except (TimeoutError, urllib.error.URLError):
        # TFE notes activity before forwarding; read timeout is normal success
        return True, "TFE tunnel ping recorded (activity refreshed)"
    except Exception as e:
        return False, str(e)


def check_hf_metrics():
    try:
        headers = {"User-Agent": "AzureRelay/1.0", "Authorization": f"Bearer {HF_TOKEN}"}
        req = urllib.request.Request(HF_METRICS_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
            update = data.get("update", 0)
            turns = data.get("total_turns", 0)
            fps = data.get("fps", 0)
            val_loss = data.get("val_loss", 0)
            log(f"[HF Checkpoint] Update: {update} | Turns: {turns:,} | Loss: {val_loss:.4f} | FPS: {fps:.1f}")
    except Exception as e:
        log(f"[HF Checkpoint Check Failed]: {e}")


def main():
    if not all([CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN, ENDPOINT]):
        log(
            "Error: Required environment variables (COLAB_CLIENT_ID, COLAB_CLIENT_SECRET, "
            "COLAB_REFRESH_TOKEN, COLAB_ENDPOINT) must be set."
        )
        return
    log(f"Starting Colab 24/7 Keep-Alive Relay on Azure VM for endpoint: {ENDPOINT}")
    ping_count = 0
    # Initial metrics check
    check_hf_metrics()
    while True:
        try:
            ok, msg = ping_keep_alive()
            ping_count += 1
            log(f"Keep-alive ping #{ping_count}: {msg}")

            if ping_count % 5 == 0:
                check_hf_metrics()

        except Exception as e:
            log(f"Error in relay loop: {e}")

        time.sleep(60)


if __name__ == "__main__":
    main()
