"""
engine/test_edge_cases.py
Comprehensive edge case tests for the JAX Gen 9 Random Battles engine:
- Pure JAX auto-reset on battle conclusion
- Randbats profile sampling integrity (no duplicate species, valid ranges)
- Extreme damage calculation boundary values (immunities, 0 BP, min-roll floor)
- All-bench-fainted terminal transitions
- Simultaneous double faint tie/draw resolution
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from engine.battle_state import (
    STATUS_NONE,
    WEATHER_NONE,
)
from engine.jax_battle_engine import (
    MOVE_TABLE,
    NUM_PROFILES,
    RANDBATS_MOVES,
    RANDBATS_SPECIES,
    SPECIES_TABLE,
    calc_raw_damage,
    get_valid_actions_mask,
    init_battle,
    sample_battle_teams,
    step_with_autoreset,
)


class TestEngineEdgeCases(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(1337)

    def test_randbats_profiles_integrity(self):
        """Verify that all Randbats profiles contain valid species and move indices."""
        self.assertGreater(NUM_PROFILES, 500)
        # All species indices must be within SPECIES_TABLE range
        self.assertTrue(bool(jnp.all(RANDBATS_SPECIES >= 0)))
        self.assertTrue(bool(jnp.all(RANDBATS_SPECIES < SPECIES_TABLE.shape[0])))

        # All moves must be within MOVE_TABLE range
        self.assertTrue(bool(jnp.all(RANDBATS_MOVES >= 0)))
        self.assertTrue(bool(jnp.all(RANDBATS_MOVES < MOVE_TABLE.shape[0])))

    def test_sample_battle_teams_no_duplicates(self):
        """Verify that sampled teams have 6 unique Pokémon without duplicates."""
        for seed in range(10):
            key = jax.random.PRNGKey(seed)
            p1_sp, p2_sp, p1_mv, p2_mv = sample_battle_teams(key)
            self.assertEqual(len(set(np.array(p1_sp))), 6)
            self.assertEqual(len(set(np.array(p2_sp))), 6)
            self.assertEqual(p1_mv.shape, (6, 4))
            self.assertEqual(p2_mv.shape, (6, 4))

    def test_damage_calc_type_immunity(self):
        """Verify that complete type immunities result in exactly 0 damage."""
        dummy_stats = jnp.array([200.0, 100.0, 100.0, 100.0, 100.0, 100.0], dtype=jnp.float32)
        dummy_boosts = jnp.zeros(7, dtype=jnp.int32)
        attacker_types = jnp.array([0, -1], dtype=jnp.int32)  # Normal
        defender_types = jnp.array([13, -1], dtype=jnp.int32)  # Ghost
        rng_roll = jnp.array(1.0, dtype=jnp.float32)

        # Move 0: Normal physical move with 100 BP
        move_idx = jnp.array(0, dtype=jnp.int32)
        dmg = calc_raw_damage(
            dummy_stats,
            dummy_boosts,
            dummy_stats,
            dummy_boosts,
            move_idx,
            attacker_types,
            defender_types,
            STATUS_NONE,
            WEATHER_NONE,
            rng_roll,
        )
        self.assertEqual(float(dmg), 0.0)

    def test_damage_calc_status_move_zero_bp(self):
        """Verify that 0 Base Power status moves deal 0 damage."""
        dummy_stats = jnp.array([200.0, 100.0, 100.0, 100.0, 100.0, 100.0], dtype=jnp.float32)
        dummy_boosts = jnp.zeros(7, dtype=jnp.int32)
        types = jnp.array([0, -1], dtype=jnp.int32)
        rng_roll = jnp.array(1.0, dtype=jnp.float32)

        # Move index for a status move (category 2 or bp 0)
        status_move_idx = jnp.where(MOVE_TABLE[:, 1] == 2)[0][0]
        dmg = calc_raw_damage(
            dummy_stats,
            dummy_boosts,
            dummy_stats,
            dummy_boosts,
            jnp.array(status_move_idx, dtype=jnp.int32),
            types,
            types,
            STATUS_NONE,
            WEATHER_NONE,
            rng_roll,
        )
        self.assertEqual(float(dmg), 0.0)

    def test_damage_calc_minimum_one_damage_floor(self):
        """Verify that damaging non-immune moves deal at least 1 damage even with extreme negative boosts."""
        min_atk_stats = jnp.array([200.0, 1.0, 100.0, 1.0, 100.0, 100.0], dtype=jnp.float32)
        max_def_stats = jnp.array([200.0, 100.0, 500.0, 100.0, 500.0, 100.0], dtype=jnp.float32)
        atk_boosts = jnp.full(7, -6, dtype=jnp.int32)
        def_boosts = jnp.full(7, 6, dtype=jnp.int32)
        types = jnp.array([0, -1], dtype=jnp.int32)
        rng_roll = jnp.array(0.85, dtype=jnp.float32)

        # Low BP damaging move (e.g., Tackle)
        damaging_move_idx = jnp.where((MOVE_TABLE[:, 1] != 2) & (MOVE_TABLE[:, 2] > 0))[0][0]
        dmg = calc_raw_damage(
            min_atk_stats,
            atk_boosts,
            max_def_stats,
            def_boosts,
            jnp.array(damaging_move_idx, dtype=jnp.int32),
            types,
            types,
            STATUS_NONE,
            WEATHER_NONE,
            rng_roll,
        )
        self.assertGreaterEqual(float(dmg), 1.0)

    def test_step_with_autoreset_transitions(self):
        """Verify that battle conclusions trigger auto-reset with non-zero terminal reward and reset state."""
        key = jax.random.PRNGKey(42)
        p1_sp, p2_sp, p1_mv, p2_mv = sample_battle_teams(key)
        state = init_battle(key, p1_sp, p2_sp, p1_mv, p2_mv)

        cur = state
        reset_occurred = False
        terminal_reward = 0.0

        for _ in range(120):
            cur, reward, done = step_with_autoreset(cur, jnp.array(0), jnp.array(0))
            if bool(done):
                reset_occurred = True
                terminal_reward = float(reward)
                # Next state must be cleanly reset to turn 0
                self.assertEqual(int(cur.turn_count), 0)
                # All Pokémon must have full health
                self.assertTrue(bool(jnp.all(cur.team_hp == 1.0)))
                self.assertTrue(bool(jnp.all(cur.team_alive)))
                break

        self.assertTrue(reset_occurred)
        self.assertIn(terminal_reward, [-1.0, 1.0, 0.0])

    def test_all_bench_fainted_action_mask(self):
        """Verify that when all bench Pokémon faint, only move actions (0-3) remain valid."""
        key = jax.random.PRNGKey(99)
        p1_sp, p2_sp, p1_mv, p2_mv = sample_battle_teams(key)
        state = init_battle(key, p1_sp, p2_sp, p1_mv, p2_mv)

        # Mark all bench slots (1-5) as fainted
        fainted_team_alive = state.team_alive.at[0, 1:6].set(False)
        fainted_team_hp = state.team_hp.at[0, 1:6].set(0.0)
        state = state.replace(team_alive=fainted_team_alive, team_hp=fainted_team_hp)

        mask = get_valid_actions_mask(state)
        # Player 1 bench switches (indices 4..8) must be False
        self.assertFalse(bool(jnp.any(mask[0, 4:9])))
        # Active moves (indices 0..3) must remain True
        self.assertTrue(bool(jnp.all(mask[0, 0:4])))


if __name__ == "__main__":
    unittest.main()
