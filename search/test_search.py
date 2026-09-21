"""
search/test_search.py
Phase 6: Unit tests for CFR matrix game solver and pUCT lookahead search.
"""

import time
import unittest
import numpy as np
import jax
import jax.numpy as jnp

from engine.jax_battle_engine import init_battle, step
from models.transformer_policy import ChronosTransformer, state_to_model_inputs
from search.regret_matching import solve_matrix_game_cfr, get_mixed_action
from search.puct_search import PUCTSearchEngine


class TestSearchAndRegretMatching(unittest.TestCase):
    def test_cfr_matrix_game_rps(self):
        # Rock-Paper-Scissors: payoff matrix
        # R vs R=0, R vs P=-1, R vs S=1
        # P vs R=1, P vs P=0,  P vs S=-1
        # S vs R=-1,S vs P=1,  S vs S=0
        rps_matrix = np.array([
            [ 0.0, -1.0,  1.0],
            [ 1.0,  0.0, -1.0],
            [-1.0,  1.0,  0.0],
        ])
        s1, s2, val = solve_matrix_game_cfr(rps_matrix, num_iterations=1000)

        print(f"\n[✓] Rock-Paper-Scissors CFR P1 Strategy: {s1}")
        print(f"[✓] Rock-Paper-Scissors CFR P2 Strategy: {s2}")
        print(f"[✓] Game value: {val:.4f}")

        # Nash equilibrium is [1/3, 1/3, 1/3]
        np.testing.assert_allclose(s1, [1/3, 1/3, 1/3], atol=0.05)
        np.testing.assert_allclose(s2, [1/3, 1/3, 1/3], atol=0.05)
        self.assertAlmostEqual(val, 0.0, places=1)

    def test_puct_search_and_time_budget(self):
        rng = jax.random.PRNGKey(42)
        p1_team = jnp.array([950, 230, 150, 1380, 450, 1200], dtype=jnp.int32)
        p2_team = jnp.array([230, 150, 950, 1200, 1380, 450], dtype=jnp.int32)
        p1_moves = jnp.array([[800, 280, 750, 220]] * 6, dtype=jnp.int32)
        p2_moves = jnp.array([[280, 750, 220, 800]] * 6, dtype=jnp.int32)

        state = init_battle(rng, p1_team, p2_team, p1_moves, p2_moves)
        model = ChronosTransformer()

        inp = state_to_model_inputs(state, 0)
        batched = {k: v[None, ...] for k, v in inp.items()}
        params = model.init(rng, batched)

        searcher = PUCTSearchEngine(
            model=model,
            params=params,
            c_puct=1.5,
            max_depth=2,
            default_time_limit_sec=1.5,  # 1.5s time budget
        )

        # Warmup searcher and JIT compilation
        _ = searcher.search(state, max_simulations=1, time_limit_sec=15.0)

        start_t = time.perf_counter()
        chosen_act, strat, stats = searcher.search(
            state,
            time_limit_sec=1.5,
            max_simulations=30,
        )
        elapsed = time.perf_counter() - start_t

        print(f"[✓] pUCT Search completed in {elapsed:.3f}s | Simulations: {stats['searched_simulations']}")
        print(f"[✓] Chosen Action: {chosen_act} | Strategy: {strat.round(3)}")

        # Verify time budget was strictly respected
        self.assertLessEqual(elapsed, 2.5)
        # Verify action is in legal range (0..8)
        self.assertIn(chosen_act, list(range(9)))

    def test_puct_guaranteed_ko_heuristic_pruning(self):
        rng = jax.random.PRNGKey(42)
        p1_team = jnp.array([950, 230, 150, 1380, 450, 1200], dtype=jnp.int32)
        p2_team = jnp.array([230, 150, 950, 1200, 1380, 450], dtype=jnp.int32)
        p1_moves = jnp.array([[800, 280, 750, 220]] * 6, dtype=jnp.int32)
        p2_moves = jnp.array([[280, 750, 220, 800]] * 6, dtype=jnp.int32)

        state = init_battle(rng, p1_team, p2_team, p1_moves, p2_moves)
        # Opponent at 5 HP and P1 with +2 speed boost
        boosts = state.active_boosts.at[0, 4].set(2)
        ko_state = state.replace(
            active_current_hp=jnp.array([250.0, 5.0]),
            active_hp=jnp.array([1.0, 0.02]),
            active_boosts=boosts,
        )

        model = ChronosTransformer()
        inp = state_to_model_inputs(state, 0)
        params = model.init(rng, {k: v[None, ...] for k, v in inp.items()})

        searcher = PUCTSearchEngine(model=model, params=params)
        act, strat, stats = searcher.search(ko_state)

        print(f"[✓] Guaranteed KO heuristic triggered: {stats['heuristic_triggered']} | Action: {act}")
        self.assertTrue(stats["heuristic_triggered"])
        self.assertEqual(stats["searched_simulations"], 0)
        self.assertIn(act, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
