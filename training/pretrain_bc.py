"""
Supervised Behavioral Cloning for policy initialization.
"""

import pickle
import time
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax

from models.transformer_policy import ChronosTransformer


def load_dataset(data_path: Path) -> Dict[str, np.ndarray]:
    if not data_path.exists():
        from data.replay_scraper import generate_synthetic_randbats_demonstrations, save_dataset

        dataset = generate_synthetic_randbats_demonstrations(num_samples=2500)
        save_dataset(dataset, data_path)
    return dict(np.load(data_path))


def smooth_labels(labels: jnp.ndarray, num_classes: int = 9, smoothing: float = 0.1) -> jnp.ndarray:
    one_hot = jax.nn.one_hot(labels, num_classes)
    return (1.0 - smoothing) * one_hot + (smoothing / num_classes)


def train_bc(
    data_path: Path,
    output_checkpoint_dir: Path,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    seed: int = 42,
) -> Dict:
    rng = jax.random.PRNGKey(seed)
    data = load_dataset(data_path)
    num_samples = len(data["actions"])

    model = ChronosTransformer()
    sample_inputs = {
        k: jnp.array(data[k][:1])
        for k in [
            "act_species",
            "act_types",
            "act_continuous",
            "bench_species_p1",
            "bench_cont_p1",
            "bench_species_p2",
            "bench_cont_p2",
            "active_moves",
            "move_types",
            "move_continuous",
            "field_features",
        ]
    }

    rng, init_key = jax.random.split(rng)
    params = model.init(init_key, sample_inputs)

    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
    opt_state = optimizer.init(params)

    def compute_loss(p, batch_inp, batch_act, batch_val):
        logits, _, pred_val = model.apply(p, batch_inp, deterministic=False)
        targets = smooth_labels(batch_act, num_classes=9, smoothing=0.1)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce_loss = -jnp.mean(jnp.sum(targets * log_probs, axis=-1))
        val_loss = 0.5 * jnp.mean((pred_val.squeeze(-1) - batch_val) ** 2)
        total_loss = ce_loss + val_loss

        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == batch_act)
        return total_loss, (ce_loss, val_loss, acc)

    @jax.jit
    def step_fn(p, opt_s, batch_inp, batch_act, batch_val):
        grads, (ce_loss, val_loss, acc) = jax.grad(compute_loss, has_aux=True)(p, batch_inp, batch_act, batch_val)
        updates, opt_s = optimizer.update(grads, opt_s, p)
        p = optax.apply_updates(p, updates)
        return p, opt_s, ce_loss, val_loss, acc

    indices = np.arange(num_samples)
    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        epoch_ce = 0.0
        epoch_val = 0.0
        epoch_acc = 0.0
        num_batches = num_samples // batch_size

        start_t = time.perf_counter()
        for b in range(num_batches):
            idx = indices[b * batch_size : (b + 1) * batch_size]
            batch_inp = {k: jnp.array(data[k][idx]) for k in sample_inputs.keys()}
            batch_act = jnp.array(data["actions"][idx])
            batch_val = jnp.array(data["values"][idx])

            params, opt_state, ce_l, val_l, acc = step_fn(params, opt_state, batch_inp, batch_act, batch_val)
            epoch_ce += float(ce_l)
            epoch_val += float(val_l)
            epoch_acc += float(acc)

        elapsed = time.perf_counter() - start_t
        print(
            f"Epoch {epoch:2d}/{epochs:2d} | "
            f"Loss: {epoch_ce / num_batches:.4f} | "
            f"Val Loss: {epoch_val / num_batches:.4f} | "
            f"Accuracy: {epoch_acc / num_batches * 100:.1f}% | "
            f"Time: {elapsed:.2f}s"
        )

    output_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_checkpoint_dir / "bc_checkpoint_latest.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(params, f)

    return params


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "train_data.npz"
    ckpt_dir = base_dir / "checkpoints"
    train_bc(data_path=data_path, output_checkpoint_dir=ckpt_dir, epochs=3, batch_size=64)


if __name__ == "__main__":
    main()
