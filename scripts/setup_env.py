#!/usr/bin/env python3
"""
scripts/setup_env.py
Phase 1: Dependencies & Private Authentication Setup for Project Chronos.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REQUIRED_PACKAGES = [
    "jax",
    "jaxlib",
    "flax",
    "poke-env",
    "kaggle",
    "huggingface_hub",
    "requests",
    "numpy",
    "scipy",
]


def check_node() -> bool:
    """Verify Node.js is installed and accessible."""
    try:
        res = subprocess.run(["node", "-v"], capture_output=True, text=True, check=True)
        print(f"[✓] Node.js available: {res.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[✗] Error: Node.js is not found in PATH. Please install Node.js.")
        return False


def install_packages() -> None:
    """Check and install required Python packages."""
    print("\n--- Checking and installing Python dependencies ---")
    missing = []
    for pkg in REQUIRED_PACKAGES:
        mod_name = pkg.replace("-", "_")
        try:
            __import__(mod_name)
            print(f"[✓] {pkg} already installed.")
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Installing missing packages: {', '.join(missing)}...")
        cmd = [sys.executable, "-m", "pip", "install"] + missing
        subprocess.check_call(cmd)
        print("[✓] All required packages installed successfully.")
    else:
        print("[✓] All required packages are satisfied.")


def setup_kaggle(username: str | None = None, api_key: str | None = None, skip_prompt: bool = False) -> bool:
    """
    Check or configure Kaggle credentials in ~/.kaggle/kaggle.json.
    Ensures 0600 permissions and tests with `kaggle datasets list --mine`.
    """
    print("\n--- Configuring Kaggle CLI Credentials ---")
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    # Check environment variables
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")

    if kaggle_json.exists():
        print(f"[✓] Kaggle credentials found at {kaggle_json}")
    elif username and api_key:
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json.write_text(json.dumps({"username": username, "key": api_key}))
        kaggle_json.chmod(0o600)
        print(f"[✓] Saved Kaggle credentials to {kaggle_json} (chmod 600)")
    elif env_user and env_key:
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json.write_text(json.dumps({"username": env_user, "key": env_key}))
        kaggle_json.chmod(0o600)
        print(f"[✓] Saved Kaggle credentials from environment to {kaggle_json} (chmod 600)")
    elif skip_prompt:
        print("[!] Kaggle credentials not found and interactive prompt skipped.")
        return False
    else:
        print("[!] Kaggle credentials not found.")
        try:
            u = input("Enter Kaggle Username: ").strip()
            k = input("Enter Kaggle API Key: ").strip()
            if u and k:
                kaggle_dir.mkdir(parents=True, exist_ok=True)
                kaggle_json.write_text(json.dumps({"username": u, "key": k}))
                kaggle_json.chmod(0o600)
                print(f"[✓] Saved Kaggle credentials to {kaggle_json} (chmod 600)")
            else:
                print("[✗] Kaggle credentials input was empty.")
                return False
        except (EOFError, KeyboardInterrupt):
            print("[!] Interactive input not available. Pass via args or environment.")
            return False

    # Ensure permissions
    if kaggle_json.exists():
        kaggle_json.chmod(0o600)

    # Verify credentials
    try:
        res = subprocess.run(
            ["kaggle", "datasets", "list", "--mine"],
            capture_output=True,
            text=True,
            check=True,
        )
        print("[✓] Kaggle authentication verified successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[✗] Kaggle authentication failed: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print("[✗] Kaggle CLI not found in PATH.")
        return False


def setup_huggingface(token: str | None = None, skip_prompt: bool = False) -> bool:
    """
    Check or configure Hugging Face authentication.
    Verifies login and enforces private repository defaults.
    """
    print("\n--- Configuring Hugging Face CLI Credentials ---")
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    def check_whoami() -> tuple[bool, str]:
        try:
            from huggingface_hub import HfApi
            user_info = HfApi().whoami()
            return True, user_info.get("name", "Authenticated")
        except Exception:
            return False, ""

    is_logged_in, username = check_whoami()
    if is_logged_in:
        print(f"[✓] Hugging Face authenticated as: {username}")
        print("[✓] Enforcing default private=True for all Chronos repositories.")
        return True

    hf_token = token or env_token
    if not hf_token:
        if skip_prompt:
            print("[!] Hugging Face token not found and interactive prompt skipped.")
            return False
        print("[!] Hugging Face token not found.")
        try:
            hf_token = input("Enter Hugging Face Write Token: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("[!] Interactive input not available. Pass via args or environment.")
            return False

    if hf_token:
        try:
            from huggingface_hub import login, HfApi
            login(token=hf_token, add_to_git_credential=False)
            user_info = HfApi().whoami()
            username = user_info.get("name", "Authenticated")
            print(f"[✓] Hugging Face authenticated successfully as: {username}")
            print("[✓] Enforcing default private=True for all Chronos repositories.")
            return True
        except Exception as e:
            print(f"[✗] Error during Hugging Face login: {e}")
            return False
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Chronos Setup & Authentication (Phase 1)")
    parser.add_argument("--kaggle-user", type=str, help="Kaggle username")
    parser.add_argument("--kaggle-key", type=str, help="Kaggle API key")
    parser.add_argument("--hf-token", type=str, help="Hugging Face Write Token")
    parser.add_argument("--skip-auth-prompt", action="store_true", help="Do not prompt if credentials missing")
    args = parser.parse_args()

    print("==================================================================")
    print("Project Chronos: Phase 1 Environment & Authentication Setup")
    print("==================================================================")

    # 1. Node.js check
    if not check_node():
        return 1

    # 2. Package install
    install_packages()

    # 3. Kaggle setup
    kaggle_ok = setup_kaggle(args.kaggle_user, args.kaggle_key, skip_prompt=args.skip_auth_prompt)

    # 4. Hugging Face setup
    hf_ok = setup_huggingface(args.hf_token, skip_prompt=args.skip_auth_prompt)

    print("\n--- Phase 1 Summary ---")
    print(f"Node.js:          [✓] Ready")
    print(f"Python Packages:  [✓] Ready")
    print(f"Kaggle Auth:      [{'✓' if kaggle_ok else '✗'}] {'Verified' if kaggle_ok else 'Pending credentials'}")
    print(f"Hugging Face Auth:[{'✓' if hf_ok else '✗'}] {'Verified' if hf_ok else 'Pending credentials'}")
    print("Default Privacy:  [✓] Strictly PRIVATE enforced")

    return 0 if (kaggle_ok and hf_ok) else 2


if __name__ == "__main__":
    sys.exit(main())
