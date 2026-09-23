#!/usr/bin/env python3
"""
Environment and credentials setup.
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
    try:
        subprocess.run(["node", "-v"], capture_output=True, text=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_packages() -> None:
    missing = []
    for pkg in REQUIRED_PACKAGES:
        mod_name = pkg.replace("-", "_")
        try:
            __import__(mod_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        cmd = [sys.executable, "-m", "pip", "install"] + missing
        subprocess.check_call(cmd)


def setup_kaggle(
    username: str | None = None, api_key: str | None = None, skip_prompt: bool = False
) -> bool:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    access_token = kaggle_dir / "access_token"

    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    env_token = os.environ.get("KAGGLE_API_TOKEN")

    if access_token.exists() or kaggle_json.exists():
        pass
    elif env_token:
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        access_token.write_text(env_token)
        access_token.chmod(0o600)
    elif username and api_key:
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json.write_text(json.dumps({"username": username, "key": api_key}))
        kaggle_json.chmod(0o600)
    elif env_user and env_key:
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json.write_text(json.dumps({"username": env_user, "key": env_key}))
        kaggle_json.chmod(0o600)
    elif not skip_prompt:
        try:
            u = input("Kaggle Username: ").strip()
            k = input("Kaggle API Key: ").strip()
            if u and k:
                kaggle_dir.mkdir(parents=True, exist_ok=True)
                kaggle_json.write_text(json.dumps({"username": u, "key": k}))
                kaggle_json.chmod(0o600)
            else:
                return False
        except (EOFError, KeyboardInterrupt):
            return False

    try:
        subprocess.run(
            ["kaggle", "datasets", "list", "--mine"],
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except Exception:
        return False


def setup_huggingface(token: str | None = None, skip_prompt: bool = False) -> bool:
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    def check_whoami() -> tuple[bool, str]:
        try:
            from huggingface_hub import HfApi

            user_info = HfApi().whoami()
            return True, user_info.get("name", "")
        except Exception:
            return False, ""

    is_logged_in, _ = check_whoami()
    if is_logged_in:
        return True

    hf_token = token or env_token
    if not hf_token and not skip_prompt:
        try:
            hf_token = input("Hugging Face Token: ").strip()
        except (EOFError, KeyboardInterrupt):
            return False

    if hf_token:
        try:
            from huggingface_hub import login

            login(token=hf_token, add_to_git_credential=False)
            is_logged_in, _ = check_whoami()
            return is_logged_in
        except Exception:
            return False
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Chronos environment setup")
    parser.add_argument("--kaggle-user", type=str, help="Kaggle username")
    parser.add_argument("--kaggle-key", type=str, help="Kaggle API key")
    parser.add_argument("--hf-token", type=str, help="Hugging Face token")
    parser.add_argument("--skip-auth-prompt", action="store_true", help="Skip prompts")
    args = parser.parse_args()

    if not check_node():
        print("Error: node not found in PATH")
        return 1

    install_packages()
    k_ok = setup_kaggle(
        args.kaggle_user, args.kaggle_key, skip_prompt=args.skip_auth_prompt
    )
    hf_ok = setup_huggingface(args.hf_token, skip_prompt=args.skip_auth_prompt)

    return 0 if (k_ok and hf_ok) else 2


if __name__ == "__main__":
    sys.exit(main())
