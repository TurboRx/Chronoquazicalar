"""
training/selfplay_ppo.py
Phase 5: PPO Self-Play Reinforcement Learning inside JAX accelerator memory.
Features:
- High-throughput parallel environment rollouts via jax.vmap
- Generalized Advantage Estimation (GAE)
- PPO clipped surrogate objective with entropy regularization
- Automated checkpointing every 30 minutes and auto-resume
- Kaggle metadata generator with is_private=True, enable_gpu=True
"""

import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from engine.battle_state import BattleState
from engine.jax_battle_engine import (
    init_battle,
    get_valid_actions_mask,
    batch_step,
    SPECIES_TABLE,
    MOVE_TABLE,
)
from models.transformer_policy import (
    ChronosTransformer,
    batch_state_to_model_inputs,
)


class RolloutBuffer(NamedTuple):
    states: Dict[str, jnp.ndarray]
    actions: jnp.ndarray
    log_probs: jnp.ndarray
    rewards: jnp.ndarray
    values: jnp.ndarray
    dones: jnp.ndarray
    valid_masks: jnp.ndarray


def prepare_kaggle_kernel_metadata(kernel_dir: Path, kernel_slug: str = "project-chronos-selfplay-ppo") -> None:
    """
    Creates kernel-metadata.json configured for private execution with GPU/TPU enabled.
    """
    metadata = {
        "id": f"{kernel_slug}",
        "title": "Project Chronos PPO Self-Play Training",
        "code_file": "training/selfplay_ppo.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,  # Strictly PRIVATE
        "enable_gpu": True,   # GPU accelerator enabled
        "enable_internet": True,
        "dataset_sources": ["project-chronos-randbats-data"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    meta_file = kernel_dir / "kernel-metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[✓] Created private Kaggle kernel configuration at {meta_file}")


def compute_gae(
    rewards: jnp.ndarray,    # (T, B)
    values: jnp.ndarray,     # (T, B)
    next_value: jnp.ndarray, # (B,)
    dones: jnp.ndarray,      # (T, B)
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes Generalized Advantage Estimation (GAE).
    """
    T = rewards.shape[0]
    advantages = []
    gae = jnp.zeros_like(next_value)

    for t in reversed(range(T)):
        non_terminal = 1.0 - dones[t].astype(jnp.float32)
        v_next = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * v_next * non_terminal - values[t]
        gae = delta + gamma * lam * non_terminal * gae
        advantages.insert(0, gae)

    advantages = jnp.stack(advantages, axis=0)  # (T, B)
    returns = advantages + values
    return advantages, returns


def train_ppo_selfplay(
    num_envs: int = 512,
    rollout_len: int = 16,
    total_updates: int = 10,
    lr: float = 2.5e-4,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    checkpoint_interval_sec: float = 1800.0,  # 30 minutes
    checkpoint_dir: Optional[Path] = None,
    seed: int = 42,
) -> Dict:
    """
    Runs PPO self-play training loop with auto-checkpointing and resume.
    """
    print(f"\n=======================================================")
    print(f"Project Chronos: Phase 5 PPO Self-Play Training")
    print(f"=======================================================")
    print(f"Parallel Environments: {num_envs} | Rollout Length: {rollout_len}")

    if checkpoint_dir is None:
        checkpoint_dir = Path("/kaggle/working") if os.path.exists("/kaggle/working") else Path("./checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_file = checkpoint_dir / "checkpoint_latest.pkl"

    rng = jax.random.PRNGKey(seed)
    model = ChronosTransformer()

    # Sample species and moves for initial teams
    rng, k1, k2, k3, k4 = jax.random.split(rng, 5)
    sample_species = jnp.array([950, 230, 150, 1380, 450, 1200], dtype=jnp.int32)
    sample_moves = jnp.array([[100, 200, 300, 400]] * 6, dtype=jnp.int32)

    p1_teams = jnp.tile(sample_species, (num_envs, 1))
    p2_teams = jnp.tile(sample_species, (num_envs, 1))
    p1_mvs = jnp.tile(sample_moves, (num_envs, 1, 1))
    p2_mvs = jnp.tile(sample_moves, (num_envs, 1, 1))

    subkeys = jax.random.split(rng, num_envs)
    init_vmap = jax.vmap(init_battle, in_axes=(0, 0, 0, 0, 0))
    states = init_vmap(subkeys, p1_teams, p2_teams, p1_mvs, p2_mvs)

    # Initialize model parameters
    sample_inp = batch_state_to_model_inputs(states[:1], 0)
    rng, init_key = jax.random.split(rng)
    params = model.init(init_key, sample_inp)

    # Auto-resume from prior checkpoint if exists
    if latest_ckpt_file.exists():
        print(f"[✓] Found existing checkpoint at {latest_ckpt_file}. Auto-resuming...")
        with open(latest_ckpt_file, "rb") as f:
            params = pickle.load(f)
    elif (checkpoint_dir / "bc_checkpoint_latest.pkl").exists():
        bc_ckpt = checkpoint_dir / "bc_checkpoint_latest.pkl"
        print(f"[✓] Initializing PPO weights from BC checkpoint at {bc_ckpt}...")
        with open(bc_ckpt, "rb") as f:
            params = pickle.load(f)

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adamw(learning_rate=lr, weight_decay=1e-4),
    )
    opt_state = optimizer.init(params)

    # PPO Loss function
    def ppo_loss(p, batch_inp, batch_acts, old_log_probs, advantages, returns, valid_masks):
        logits, probs, values = model.apply(p, batch_inp, valid_mask=valid_masks, deterministic=False)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        action_log_probs = jnp.take_along_axis(log_probs, batch_acts[:, None], axis=-1).squeeze(-1)

        # Ratio
        ratio = jnp.exp(action_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        # Value loss
        val_loss = 0.5 * jnp.mean((values.squeeze(-1) - returns) ** 2)

        # Entropy bonus
        entropy = -jnp.mean(jnp.sum(probs * jnp.log(jnp.maximum(probs, 1e-8)), axis=-1))
        total_loss = policy_loss + 0.5 * val_loss - entropy_coef * entropy

        return total_loss, (policy_loss, val_loss, entropy)

    @jax.jit
    def update_step(p, opt_s, batch_inp, batch_acts, old_log_p, advs, rets, v_masks):
        grads, (pol_l, val_l, ent) = jax.grad(ppo_loss, has_aux=True)(
            p, batch_inp, batch_acts, old_log_p, advs, rets, v_masks
        )
        updates, opt_s = optimizer.update(grads, opt_s, p)
        p = optax.apply_updates(p, updates)
        return p, opt_s, pol_l, val_l, ent

    # Main PPO Update Loop
    last_checkpoint_time = time.time()
    for update in range(1, total_updates + 1):
        start_t = time.perf_counter()

        # 1. Collect Rollout
        buf_states_list = []
        buf_actions = []
        buf_log_probs = []
        buf_rewards = []
        buf_values = []
        buf_dones = []
        buf_masks = []

        cur_states = states
        for step_idx in range(rollout_len):
            rng, akey1, akey2 = jax.random.split(rng, 3)
            # Model inputs for P1
            inp_p1 = batch_state_to_model_inputs(cur_states, 0)
            masks_p1 = jax.vmap(get_valid_actions_mask)(cur_states)[:, 0]  # (B, 9)

            logits_p1, probs_p1, vals_p1 = model.apply(params, inp_p1, valid_mask=masks_p1)
            actions_p1 = jax.random.categorical(akey1, logits_p1)

            # Self-play: opponent uses same policy
            inp_p2 = batch_state_to_model_inputs(cur_states, 1)
            masks_p2 = jax.vmap(get_valid_actions_mask)(cur_states)[:, 1]
            logits_p2, _, _ = model.apply(params, inp_p2, valid_mask=masks_p2)
            actions_p2 = jax.random.categorical(akey2, logits_p2)

            # Step environment
            next_states, rewards, dones = batch_step(cur_states, actions_p1, actions_p2)

            log_p = jax.nn.log_softmax(logits_p1, axis=-1)
            act_log_p = jnp.take_along_axis(log_p, actions_p1[:, None], axis=-1).squeeze(-1)

            buf_states_list.append(inp_p1)
            buf_actions.append(actions_p1)
            buf_log_probs.append(act_log_p)
            buf_rewards.append(rewards)
            buf_values.append(vals_p1.squeeze(-1))
            buf_dones.append(dones)
            buf_masks.append(masks_p1)

            cur_states = next_states

        states = cur_states

        # 2. Compute GAE
        last_inp = batch_state_to_model_inputs(states, 0)
        _, _, next_v = model.apply(params, last_inp)
        rewards_arr = jnp.stack(buf_rewards, axis=0)
        values_arr = jnp.stack(buf_values, axis=0)
        dones_arr = jnp.stack(buf_dones, axis=0)

        advantages, returns = compute_gae(rewards_arr, values_arr, next_v.squeeze(-1), dones_arr)
        # Normalize advantages
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

        # 3. Flatten and Update
        flat_acts = jnp.concatenate(buf_actions, axis=0)
        flat_log_p = jnp.concatenate(buf_log_probs, axis=0)
        flat_adv = advantages.reshape(-1)
        flat_ret = returns.reshape(-1)
        flat_masks = jnp.concatenate(buf_masks, axis=0)

        flat_inp = {k: jnp.concatenate([s[k] for s in buf_states_list], axis=0) for k in buf_states_list[0].keys()}

        params, opt_state, pol_l, val_l, ent = update_step(
            params, opt_state, flat_inp, flat_acts, flat_log_p, flat_adv, flat_ret, flat_masks
        )

        elapsed = time.perf_counter() - start_t
        fps = (num_envs * rollout_len) / elapsed
        print(
            f"Update {update:2d}/{total_updates:2d} | "
            f"Pol Loss: {float(pol_l):.4f} | "
            f"Val Loss: {float(val_l):.4f} | "
            f"Entropy: {float(ent):.4f} | "
            f"FPS: {fps:8,.1f} | Time: {elapsed:.2f}s"
        )

        # Automated Checkpointing every 30 minutes
        current_time = time.time()
        if current_time - last_checkpoint_time >= checkpoint_interval_sec or update == total_updates:
            with open(latest_ckpt_file, "wb") as f:
                pickle.dump(params, f)
            print(f"[✓] Checkpoint automatically saved to {latest_ckpt_file}")
            last_checkpoint_time = current_time

    return params


def main():
    base_dir = Path(__file__).resolve().parent.parent
    prepare_kaggle_kernel_metadata(base_dir)
    train_ppo_selfplay(num_envs=64, rollout_len=8, total_updates=3)


if __name__ == "__main__":
    main()
