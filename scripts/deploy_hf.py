"""
scripts/deploy_hf.py
Phase 7: Programmatic upload of Project Chronos trained model artifacts,
mechanics tables, and configurations to a strictly private Hugging Face repository.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi


def deploy_to_huggingface(
    repo_id: str,
    checkpoint_path: Path,
    config_path: Optional[Path] = None,
    token: Optional[str] = None,
) -> bool:
    """
    Creates or updates a private Hugging Face Model repository
    and uploads checkpoint weights and configs.
    """
    print(f"\n=======================================================")
    print(f"Project Chronos: Phase 7 Hugging Face Private Deployment")
    print(f"=======================================================")
    print(f"Target Repo: {repo_id}")
    print(f"Enforcing: private=True (Strictly Private)")

    hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("[!] HF_TOKEN not found in environment or arguments. Cannot upload.")
        return False

    api = HfApi(token=hf_token)

    try:
        # 1. Create strictly private repository
        print(f"Creating/verifying private repository: {repo_id}...")
        api.create_repo(repo_id=repo_id, private=True, exist_ok=True, repo_type="model")
        print(f"[✓] Private repository verified: https://huggingface.co/{repo_id}")

        # 2. Upload Checkpoint
        if checkpoint_path.exists():
            print(f"Uploading checkpoint {checkpoint_path.name}...")
            api.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=checkpoint_path.name,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"[✓] Checkpoint uploaded successfully.")
        else:
            print(f"[!] Warning: Checkpoint {checkpoint_path} does not exist.")

        # 3. Upload Architecture Config
        model_config = {
            "model_type": "chronos_transformer",
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 6,
            "d_ff": 1024,
            "num_actions": 9,
            "format": "gen9randombattle",
            "private": True,
        }
        config_file = checkpoint_path.parent / "config.json"
        with open(config_file, "w") as f:
            json.dump(model_config, f, indent=2)

        api.upload_file(
            path_or_fileobj=str(config_file),
            path_in_repo="config.json",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"[✓] Model config uploaded.")

        # 4. Upload Mechanics Data Tables
        mechanics_path = Path(__file__).resolve().parent.parent / "engine" / "data" / "mechanics_tables.npz"
        if mechanics_path.exists():
            api.upload_file(
                path_or_fileobj=str(mechanics_path),
                path_in_repo="mechanics_tables.npz",
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"[✓] Mechanics tables uploaded.")

        print(f"\n[✓] All artifacts successfully deployed to private repo https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        print(f"[✗] Error during Hugging Face deployment: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Deploy Chronos model to private HF repo")
    parser.add_argument("--repo-id", type=str, default="chronos-gen9-randbats", help="Hugging Face repo ID (e.g. username/repo)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/bc_checkpoint_latest.pkl", help="Path to checkpoint file")
    parser.add_argument("--token", type=str, help="Hugging Face write token")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    deploy_to_huggingface(repo_id=args.repo_id, checkpoint_path=ckpt_path, token=args.token)


if __name__ == "__main__":
    main()
