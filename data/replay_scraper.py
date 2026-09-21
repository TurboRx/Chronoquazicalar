"""
Replay scraper, tokenizer, and dataset utilities for Gen 9 Random Battles.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from engine.battle_state import BattleState
from engine.jax_battle_engine import init_battle, get_valid_actions_mask, SPECIES_TABLE, MOVE_TABLE
from models.transformer_policy import state_to_model_inputs
import jax
import jax.numpy as jnp

REPLAY_SEARCH_URL = "https://replay.pokemonshowdown.com/search.json"
REPLAY_BASE_URL = "https://replay.pokemonshowdown.com"


def fetch_high_ladder_replays(format_id: str = "gen9randombattle", min_elo: int = 1600, max_pages: int = 5) -> List[str]:
    replay_ids = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for page in range(1, max_pages + 1):
        try:
            params = {"format": format_id, "page": page}
            resp = requests.get(REPLAY_SEARCH_URL, params=params, headers=headers, timeout=10)
            if resp.status_code != 200:
                break

            data = resp.json()
            if not data:
                break

            for item in data:
                rating = item.get("rating") or 0
                if rating >= min_elo or min_elo == 0:
                    replay_ids.append(item["id"])

            time.sleep(0.5)
        except Exception:
            break

    return replay_ids


def generate_synthetic_randbats_demonstrations(num_samples: int = 2000) -> Dict[str, np.ndarray]:
    rng = jax.random.PRNGKey(777)

    sample_species = jnp.array([
        950, 230, 150, 1380, 450, 1200, 300, 400, 500, 600, 700, 800, 900, 1000
    ], dtype=jnp.int32)
    sample_moves = jnp.array([
        100, 200, 300, 400, 500, 600, 700, 800
    ], dtype=jnp.int32)

    all_inputs: Dict[str, List[np.ndarray]] = {
        "act_species": [],
        "act_types": [],
        "act_continuous": [],
        "bench_species_p1": [],
        "bench_cont_p1": [],
        "bench_species_p2": [],
        "bench_cont_p2": [],
        "active_moves": [],
        "move_types": [],
        "move_continuous": [],
        "field_features": [],
    }
    all_actions = []
    all_values = []

    for i in range(num_samples):
        rng, k1, k2, k3, k4 = jax.random.split(rng, 5)
        p1_team = jax.random.choice(k1, sample_species, shape=(6,), replace=False)
        p2_team = jax.random.choice(k2, sample_species, shape=(6,), replace=False)
        p1_mvs = jax.random.choice(k3, sample_moves, shape=(6, 4), replace=True)
        p2_mvs = jax.random.choice(k4, sample_moves, shape=(6, 4), replace=True)

        state = init_battle(rng, p1_team, p2_team, p1_mvs, p2_mvs)

        rng, khp1, khp2 = jax.random.split(rng, 3)
        hp1 = float(jax.random.uniform(khp1, shape=(), minval=0.1, maxval=1.0))
        hp2 = float(jax.random.uniform(khp2, shape=(), minval=0.1, maxval=1.0))
        state = state.replace(
            active_hp=jnp.array([hp1, hp2], dtype=jnp.float32),
            active_current_hp=jnp.array([hp1 * 250.0, hp2 * 250.0], dtype=jnp.float32),
        )

        inp = state_to_model_inputs(state, perspective_player=0)
        for k, v in inp.items():
            all_inputs[k].append(np.array(v))

        mask = np.array(get_valid_actions_mask(state)[0])
        valid_indices = np.where(mask)[0]

        if hp1 > 0.3 or np.all(~mask[4:]):
            action = int(np.random.choice(valid_indices[valid_indices < 4]))
        else:
            action = int(np.random.choice(valid_indices))

        all_actions.append(action)
        all_values.append(float(np.clip(hp1 - hp2, -1.0, 1.0)))

    dataset = {k: np.stack(v, axis=0) for k, v in all_inputs.items()}
    dataset["actions"] = np.array(all_actions, dtype=np.int32)
    dataset["values"] = np.array(all_values, dtype=np.float32)

    return dataset


def save_dataset(dataset: Dict[str, np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **dataset)


def prepare_private_kaggle_dataset(data_dir: Path, dataset_slug: str = "project-chronos-randbats-data") -> bool:
    metadata = {
        "title": "Project Chronos Gen 9 Randbats Dataset",
        "id": f"{dataset_slug}",
        "licenses": [{"name": "private"}],
        "isPrivate": True,
    }
    meta_path = data_dir / "dataset-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    try:
        res = subprocess.run(["kaggle", "datasets", "list", "--mine"], capture_output=True, text=True)
        if res.returncode != 0:
            return False

        cmd = ["kaggle", "datasets", "create", "-p", str(data_dir), "-u"]
        upload_res = subprocess.run(cmd, capture_output=True, text=True)
        if upload_res.returncode == 0:
            return True
        else:
            cmd_v = ["kaggle", "datasets", "version", "-p", str(data_dir), "-m", "Updated dataset"]
            upload_v = subprocess.run(cmd_v, capture_output=True, text=True)
            return upload_v.returncode == 0
    except FileNotFoundError:
        return False


def main():
    data_dir = Path(__file__).resolve().parent
    out_file = data_dir / "train_data.npz"

    _ = fetch_high_ladder_replays(format_id="gen9randombattle", min_elo=1600, max_pages=2)
    dataset = generate_synthetic_randbats_demonstrations(num_samples=2500)
    save_dataset(dataset, out_file)
    prepare_private_kaggle_dataset(data_dir)


if __name__ == "__main__":
    main()
