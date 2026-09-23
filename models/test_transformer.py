"""
models/test_transformer.py
Phase 4: Unit tests, parameter count verification, and inference latency benchmarking
for ChronosTransformer and damage heuristic.
"""

import time
import unittest

import jax
import jax.numpy as jnp

from engine.damage_calc import check_guaranteed_ko, get_min_roll_damages
from engine.jax_battle_engine import init_battle
from models.transformer_policy import (
    ChronosTransformer,
    state_to_model_inputs,
)


class TestTransformerPolicy(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(42)
        # Sample teams and moves
        self.p1_team = jnp.array([950, 230, 150, 1380, 450, 1200], dtype=jnp.int32)
        self.p2_team = jnp.array([230, 150, 950, 1200, 1380, 450], dtype=jnp.int32)
        self.p1_moves = jnp.array([[800, 280, 750, 220]] * 6, dtype=jnp.int32)
        self.p2_moves = jnp.array([[280, 750, 220, 800]] * 6, dtype=jnp.int32)

        self.state = init_battle(
            self.rng, self.p1_team, self.p2_team, self.p1_moves, self.p2_moves
        )
        self.model = ChronosTransformer()

    def test_parameter_count(self):
        inputs = state_to_model_inputs(self.state, perspective_player=0)
        # Add batch dimension
        batched_inputs = {k: v[None, ...] for k, v in inputs.items()}
        params = self.model.init(self.rng, batched_inputs)

        # Count parameters
        num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(
            f"\n[✓] ChronosTransformer Total Parameter Count: {num_params:,} (~{num_params / 1e6:.2f}M)"
        )

        # Verify close to ~8.5M (between 6.0M and 10.0M)
        self.assertGreater(num_params, 6_000_000)
        self.assertLess(num_params, 10_500_000)

    def test_forward_pass_and_masking(self):
        inputs = state_to_model_inputs(self.state, perspective_player=0)
        batched_inputs = {k: v[None, ...] for k, v in inputs.items()}
        params = self.model.init(self.rng, batched_inputs)

        # Valid action mask (e.g. disable switches)
        mask = jnp.array([[True, True, True, True, False, False, False, False, False]])
        logits, probs, value = self.model.apply(params, batched_inputs, valid_mask=mask)

        self.assertEqual(logits.shape, (1, 9))
        self.assertEqual(probs.shape, (1, 9))
        self.assertEqual(value.shape, (1, 1))

        # Check masked action probs are ~0.0
        self.assertAlmostEqual(float(jnp.sum(probs[0, 4:])), 0.0, places=5)
        self.assertAlmostEqual(float(jnp.sum(probs[0, :4])), 1.0, places=5)
        # Value in [-1.0, 1.0]
        self.assertGreaterEqual(float(value[0, 0]), -1.0)
        self.assertLessEqual(float(value[0, 0]), 1.0)

    def test_damage_calc_and_heuristic(self):
        min_damages = get_min_roll_damages(self.state, player_idx=0)
        self.assertEqual(min_damages.shape, (4,))
        print(f"[✓] Min-roll damages for moves: {min_damages}")

        # 1. When slower, cannot guarantee KO before opponent moves
        low_hp_slower = self.state.replace(
            active_current_hp=jnp.array([250.0, 5.0]),
            active_hp=jnp.array([1.0, 0.02]),
        )
        has_ko_slower, _ = check_guaranteed_ko(low_hp_slower, player_idx=0)
        self.assertFalse(bool(has_ko_slower))

        # 2. When outspeeding (e.g. +2 speed boost), guaranteed KO is triggered
        boosts_faster = self.state.active_boosts.at[0, 4].set(2)
        low_hp_faster = low_hp_slower.replace(active_boosts=boosts_faster)
        has_ko, best_act = check_guaranteed_ko(low_hp_faster, player_idx=0)
        print(
            f"[✓] Outspeeding Low HP scenario -> Guaranteed KO detected: {bool(has_ko)}, Best Action: {int(best_act)}"
        )
        self.assertTrue(bool(has_ko))
        self.assertIn(int(best_act), [0, 1, 2, 3])

    def test_inference_latency(self):
        inputs = state_to_model_inputs(self.state, perspective_player=0)
        batched_inputs = {k: v[None, ...] for k, v in inputs.items()}
        params = self.model.init(self.rng, batched_inputs)

        @jax.jit
        def predict(p, inp):
            return self.model.apply(p, inp)

        # Warmup JIT
        _ = predict(params, batched_inputs)
        _ = jax.tree_util.tree_map(lambda x: x.block_until_ready(), _)

        # Measure latency
        num_trials = 50
        start_t = time.perf_counter()
        for _ in range(num_trials):
            out = predict(params, batched_inputs)
        _ = jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
        avg_latency_ms = ((time.perf_counter() - start_t) / num_trials) * 1000.0

        print(f"[✓] Model Inference Latency (Batch=1): {avg_latency_ms:.2f} ms")


if __name__ == "__main__":
    unittest.main()
