"""
Uploads model checkpoints, mechanics tables, and configurations to Hugging Face.
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
    commit_message: Optional[str] = None,
    commit_description: Optional[str] = None,
) -> bool:
    hf_token = (
        token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not hf_token:
        try:
            from huggingface_hub import get_token

            hf_token = get_token()
        except Exception:
            pass
    if not hf_token:
        print("Error: HF_TOKEN not found in environment or arguments")
        return False

    api = HfApi(token=hf_token)
    msg = commit_message or f"feat(weights): update {checkpoint_path.name}"
    desc = (
        commit_description
        or "Chronos Transformer policy & value network for Gen 9 Random Battles (PPO self-play in JAX)."
    )

    try:
        api.create_repo(repo_id=repo_id, private=True, exist_ok=True, repo_type="model")

        if checkpoint_path.exists():
            api.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=checkpoint_path.name,
                repo_id=repo_id,
                repo_type="model",
                commit_message=msg,
                commit_description=desc,
            )

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

        mechanics_path = (
            Path(__file__).resolve().parent.parent
            / "engine"
            / "data"
            / "mechanics_tables.npz"
        )
        if mechanics_path.exists():
            api.upload_file(
                path_or_fileobj=str(mechanics_path),
                path_in_repo="mechanics_tables.npz",
                repo_id=repo_id,
                repo_type="model",
            )

        readme_content = """---
license: mit
pipeline_tag: reinforcement-learning
tags:
- pokemon-showdown
- jax
- reinforcement-learning
---

# Chronos (Gen 9 Random Battles)

Transformer policy and value network (~8.5M parameters) for Pokémon Showdown Gen 9 Random Battles.

## Architecture
- Architecture: Non-Causal Transformer Encoder
- Dimension (`d_model`): 256
- Attention Heads: 8
- Encoder Layers: 6
- Feed-Forward (`d_ff`): 1024
- Action Space: 9 discrete actions (4 moves + 5 switches)
"""
        readme_file = checkpoint_path.parent / "README.md"
        with open(readme_file, "w") as f:
            f.write(readme_content)

        api.upload_file(
            path_or_fileobj=str(readme_file),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )

        print(f"Deployed artifacts to https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        print(f"Deployment error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Deploy checkpoint to Hugging Face")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="chronos-gen9-randbats",
        help="Repo ID (e.g. username/repo)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/bc_checkpoint_latest.pkl",
        help="Checkpoint path",
    )
    parser.add_argument("--token", type=str, help="Hugging Face token")
    parser.add_argument("--message", type=str, help="Commit message")
    parser.add_argument("--description", type=str, help="Commit description")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    deploy_to_huggingface(
        repo_id=args.repo_id,
        checkpoint_path=ckpt_path,
        token=args.token,
        commit_message=args.message,
        commit_description=args.description,
    )


if __name__ == "__main__":
    main()
