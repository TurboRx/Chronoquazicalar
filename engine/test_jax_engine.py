"""
engine/test_jax_engine.py
Unit tests and throughput benchmarking for pure JAX Gen 9 battle engine.
"""

import time
import unittest

import jax
import jax.numpy as jnp

from engine.jax_battle_engine import (
    batch_step,
    get_valid_actions_mask,
    init_battle,
    step,
)


class TestJaxBattleEngine(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(42)
        # Sample teams (Pikachu=950, Charizard=230, Blastoise=150, Venusaur=1380, Gengar=450, Snorlax=1200)
        self.p1_team = jnp.array([950, 230, 150, 1380, 450, 1200], dtype=jnp.int32)
        self.p2_team = jnp.array([230, 150, 950, 1200, 1380, 450], dtype=jnp.int32)
        # Moves (Thunderbolt=800, Flamethrower=280, Surf=750, Energy Ball=220)
        self.p1_moves = jnp.full((6, 4), 100, dtype=jnp.int32)
        self.p2_moves = jnp.full((6, 4), 100, dtype=jnp.int32)

    def test_init_battle(self):
        state = init_battle(self.rng, self.p1_team, self.p2_team, self.p1_moves, self.p2_moves)
        self.assertEqual(state.active_hp.shape, (2,))
        self.assertEqual(state.team_alive.shape, (2, 6))
        self.assertTrue(bool(jnp.all(state.active_hp == 1.0)))
        self.assertFalse(bool(state.done))

    def test_valid_actions_mask(self):
        state = init_battle(self.rng, self.p1_team, self.p2_team, self.p1_moves, self.p2_moves)
        mask = get_valid_actions_mask(state)
        self.assertEqual(mask.shape, (2, 9))
        # Initially all 4 moves and 5 switches should be legal
        self.assertTrue(bool(jnp.all(mask)))

    def test_single_step(self):
        state = init_battle(self.rng, self.p1_team, self.p2_team, self.p1_moves, self.p2_moves)
        # Both choose move 0
        next_state, reward, done = step(state, jnp.array(0), jnp.array(0))
        self.assertEqual(next_state.turn_count, 1)
        self.assertTrue(bool(jnp.any(next_state.active_hp < 1.0)))

    def test_benchmark_throughput(self):
        devices = jax.devices()
        is_cpu = all(d.platform == "cpu" for d in devices)
        target_tps = 200.0 if is_cpu else 50000.0

        print(f"\n--- Benchmarking JAX Vectorized Battle Engine on {devices[0].platform.upper()} ---")

        for B in [512, 1024]:
            key = jax.random.PRNGKey(123)
            subkeys = jax.random.split(key, B)
            p1_teams = jnp.repeat(self.p1_team[None, :], B, axis=0)
            p2_teams = jnp.repeat(self.p2_team[None, :], B, axis=0)
            p1_mvs = jnp.repeat(self.p1_moves[None, :, :], B, axis=0)
            p2_mvs = jnp.repeat(self.p2_moves[None, :, :], B, axis=0)

            # Initialize batch of states
            init_vmap = jax.vmap(init_battle, in_axes=(0, 0, 0, 0, 0))
            states = init_vmap(subkeys, p1_teams, p2_teams, p1_mvs, p2_mvs)

            actions_p1 = jnp.zeros(B, dtype=jnp.int32)
            actions_p2 = jnp.zeros(B, dtype=jnp.int32)

            # Warmup JIT compilation
            _states, _rewards, _dones = batch_step(states, actions_p1, actions_p2)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), _states)

            # Benchmark batched steps
            num_iters = 10
            start_t = time.perf_counter()
            cur_states = states
            for _ in range(num_iters):
                cur_states, _, _ = batch_step(cur_states, actions_p1, actions_p2)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), cur_states)
            elapsed = time.perf_counter() - start_t

            total_turns = B * num_iters
            turns_per_sec = total_turns / elapsed
            print(
                f"Batch Size {B:5d} | Time: {elapsed:.3f}s | Throughput: {turns_per_sec:10,.1f} turns/sec (Platform: {devices[0].platform})"
            )
            self.assertGreater(turns_per_sec, target_tps)


if __name__ == "__main__":
    unittest.main()
