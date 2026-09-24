"""
models/test_edge_cases.py
Edge-case unit tests for the Chronos Transformer architecture:
- Highly restricted action masks (only 1 legal action)
- Action probability distribution invariants (sums to 1.0, strictly non-negative)
- Value head boundedness in [-1.0, 1.0] across diverse inputs
- Batch invariance between single and batched evaluations
"""

import unittest

import jax
import jax.numpy as jnp

from engine.jax_battle_engine import init_battle, sample_battle_teams
from models.transformer_policy import (
    ChronosTransformer,
    batch_state_to_model_inputs,
    state_to_model_inputs,
)


class TestModelEdgeCases(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(42)
        self.model = ChronosTransformer()
        k1, k2 = jax.random.split(self.rng)
        p1_sp, p2_sp, p1_mv, p2_mv = sample_battle_teams(k1)
        self.state = init_battle(k2, p1_sp, p2_sp, p1_mv, p2_mv)
        self.inp = state_to_model_inputs(self.state, 0)
        self.batched_inp = batch_state_to_model_inputs(jax.tree_util.tree_map(lambda x: x[None, ...], self.state), 0)
        self.params = self.model.init(self.rng, self.batched_inp)

    def test_single_legal_action_mask(self):
        """Verify that when only 1 action is valid, its probability is 1.0 and others are 0.0."""
        for legal_act in [0, 2, 5, 8]:
            mask = jnp.zeros((1, 9), dtype=jnp.bool_)
            mask = mask.at[0, legal_act].set(True)

            logits, probs, val = self.model.apply(self.params, self.batched_inp, valid_mask=mask)
            # Valid action probability must be near 1.0
            self.assertAlmostEqual(float(probs[0, legal_act]), 1.0, places=4)
            # Other actions must be near 0.0
            other_probs = jnp.delete(probs[0], legal_act)
            self.assertTrue(bool(jnp.all(other_probs < 1e-4)))

    def test_value_head_strictly_bounded(self):
        """Verify that value head output is strictly within [-1.0, 1.0]."""
        for _ in range(5):
            self.rng, k = jax.random.split(self.rng)
            # Corrupt continuous inputs with extreme positive and negative values
            extreme_inp = {k_inp: (v * 100.0 if "cont" in k_inp else v) for k_inp, v in self.batched_inp.items()}
            _, _, val = self.model.apply(self.params, extreme_inp)
            self.assertGreaterEqual(float(val[0, 0]), -1.0)
            self.assertLessEqual(float(val[0, 0]), 1.0)

    def test_batch_invariance(self):
        """Verify that evaluating an input individually or as part of a batch produces identical logits."""
        b4_inp = {k: jnp.repeat(v, 4, axis=0) for k, v in self.batched_inp.items()}
        logits_single, _, val_single = self.model.apply(self.params, self.batched_inp)
        logits_batch, _, val_batch = self.model.apply(self.params, b4_inp)

        np_single = jnp.squeeze(logits_single, axis=0)
        for i in range(4):
            np_batch = logits_batch[i]
            diff = jnp.max(jnp.abs(np_single - np_batch))
            self.assertLess(float(diff), 1e-5)


if __name__ == "__main__":
    unittest.main()
