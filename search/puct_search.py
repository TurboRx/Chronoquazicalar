"""
search/puct_search.py
Phase 6: Polynomial Upper Confidence Trees (pUCT) Search with Regret Matching
for simultaneous turns in Pokémon Showdown.
Features:
- Lookahead search (depth 2 to 4)
- Policy prior and value network leaf evaluations
- Matrix game CFR Nash equilibrium resolution at decision nodes
- Exact damage heuristic check to prune branches
- Strict time budget cap (<= 2.5 seconds per turn)
"""

import time
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from engine.battle_state import BattleState
from engine.damage_calc import check_guaranteed_ko
from engine.jax_battle_engine import get_valid_actions_mask, step
from models.transformer_policy import ChronosTransformer, state_to_model_inputs
from search.regret_matching import get_mixed_action, solve_matrix_game_cfr


class PUCTNode:
    """A node in the simultaneous-move pUCT search tree."""

    def __init__(
        self,
        state: BattleState,
        prior_p1: np.ndarray,
        prior_p2: np.ndarray,
        valid_mask_p1: np.ndarray,
        valid_mask_p2: np.ndarray,
        value: float = 0.0,
        is_terminal: bool = False,
    ):
        self.state = state
        self.prior_p1 = prior_p1
        self.prior_p2 = prior_p2
        self.valid_mask_p1 = valid_mask_p1
        self.valid_mask_p2 = valid_mask_p2
        self.value = value
        self.is_terminal = is_terminal

        self.valid_actions_p1 = np.where(valid_mask_p1)[0]
        self.valid_actions_p2 = np.where(valid_mask_p2)[0]

        # Statistics: shape (9, 9)
        self.q_matrix = np.zeros((9, 9), dtype=np.float32)
        self.n_matrix = np.zeros((9, 9), dtype=np.int32)
        self.n_p1 = np.zeros(9, dtype=np.int32)
        self.n_p2 = np.zeros(9, dtype=np.int32)
        self.total_visits = 0

        # Children: key is (action_p1, action_p2)
        self.children: Dict[Tuple[int, int], "PUCTNode"] = {}


class PUCTSearchEngine:
    """
    pUCT Lookahead Search Engine with Regret Matching and Strict Time Budget.
    """

    def __init__(
        self,
        model: ChronosTransformer,
        params: Dict,
        c_puct: float = 1.5,
        max_depth: int = 3,
        default_time_limit_sec: float = 2.0,  # Strict cap well below Showdown timeout
    ):
        self.model = model
        self.params = params
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.default_time_limit_sec = default_time_limit_sec

    def evaluate_state(self, state: BattleState) -> Tuple[np.ndarray, np.ndarray, float]:
        """Runs the transformer model on state from both players' perspectives."""
        # P1 evaluation
        inp_p1 = state_to_model_inputs(state, perspective_player=0)
        batched_p1 = {k: v[None, ...] for k, v in inp_p1.items()}
        mask = np.array(get_valid_actions_mask(state))
        mask_p1 = jnp.array(mask[0:1])
        _, probs_p1, val_p1 = self.model.apply(self.params, batched_p1, valid_mask=mask_p1)

        # P2 evaluation
        inp_p2 = state_to_model_inputs(state, perspective_player=1)
        batched_p2 = {k: v[None, ...] for k, v in inp_p2.items()}
        mask_p2 = jnp.array(mask[1:2])
        _, probs_p2, _ = self.model.apply(self.params, batched_p2, valid_mask=mask_p2)

        return np.array(probs_p1[0]), np.array(probs_p2[0]), float(val_p1[0, 0])

    def search(
        self,
        root_state: BattleState,
        time_limit_sec: Optional[float] = None,
        max_simulations: int = 150,
        temperature: float = 0.5,
    ) -> Tuple[int, np.ndarray, Dict]:
        """
        Executes pUCT search with CFR regret matching.
        Returns:
          chosen_action: int (0..8)
          nash_strategy: np.ndarray (9,)
          search_stats: Dict
        """
        start_time = time.perf_counter()
        time_limit = time_limit_sec or self.default_time_limit_sec

        # 1. Exact Damage & Rule Heuristic Check:
        # If guaranteed min-roll KO and outspeed, return action immediately!
        has_ko, ko_action = check_guaranteed_ko(root_state, player_idx=0)
        if bool(has_ko) and int(ko_action) >= 0:
            act = int(ko_action)
            strat = np.zeros(9, dtype=np.float32)
            strat[act] = 1.0
            stats = {
                "searched_simulations": 0,
                "elapsed_sec": time.perf_counter() - start_time,
                "heuristic_triggered": True,
                "expected_value": 1.0,
            }
            return act, strat, stats

        # 2. Initialize Root Node
        probs_p1, probs_p2, val = self.evaluate_state(root_state)
        masks = np.array(get_valid_actions_mask(root_state))
        root = PUCTNode(
            state=root_state,
            prior_p1=probs_p1,
            prior_p2=probs_p2,
            valid_mask_p1=masks[0],
            valid_mask_p2=masks[1],
            value=val,
            is_terminal=bool(root_state.done),
        )

        sim_count = 0

        # 3. Main Search Loop with Strict Time Budget Cap
        while sim_count < max_simulations:
            elapsed = time.perf_counter() - start_time
            if elapsed >= time_limit:
                break

            # Selection & Rollout
            self._simulate(root, depth=0)
            sim_count += 1

        # 4. Resolve Root Decision via CFR on Accumulated Q-Matrix
        valid_acts_p1 = root.valid_actions_p1
        valid_acts_p2 = root.valid_actions_p2

        # Sub-matrix for valid actions
        payoff_submatrix = root.q_matrix[np.ix_(valid_acts_p1, valid_acts_p2)]
        chosen_action, nash_strat = get_mixed_action(
            valid_actions=valid_acts_p1,
            payoff_matrix=payoff_submatrix,
            num_iterations=500,
            temperature=temperature,
        )

        total_elapsed = time.perf_counter() - start_time
        stats = {
            "searched_simulations": sim_count,
            "elapsed_sec": total_elapsed,
            "heuristic_triggered": False,
            "expected_value": float(root.value),
            "root_visits": root.total_visits,
        }

        return chosen_action, nash_strat, stats

    def _simulate(self, node: PUCTNode, depth: int) -> float:
        """Simulates one path down the search tree."""
        if node.is_terminal or depth >= self.max_depth:
            return node.value

        # Select action pair (a1, a2) using pUCT
        a1, a2 = self._select_action_pair(node)
        pair = (a1, a2)

        if pair not in node.children:
            # Expand node
            act_p1 = jnp.array(a1, dtype=jnp.int32)
            act_p2 = jnp.array(a2, dtype=jnp.int32)
            next_state, reward, done = step(node.state, act_p1, act_p2)

            is_term = bool(done)
            if is_term:
                leaf_val = float(reward)
                p1_priors = np.zeros(9, dtype=np.float32)
                p2_priors = np.zeros(9, dtype=np.float32)
                v_masks = np.array(get_valid_actions_mask(next_state))
            else:
                p1_priors, p2_priors, leaf_val = self.evaluate_state(next_state)
                v_masks = np.array(get_valid_actions_mask(next_state))

            child = PUCTNode(
                state=next_state,
                prior_p1=p1_priors,
                prior_p2=p2_priors,
                valid_mask_p1=v_masks[0],
                valid_mask_p2=v_masks[1],
                value=leaf_val,
                is_terminal=is_term,
            )
            node.children[pair] = child
            value = leaf_val
        else:
            value = self._simulate(node.children[pair], depth + 1)

        # Backpropagation
        node.n_matrix[a1, a2] += 1
        node.n_p1[a1] += 1
        node.n_p2[a2] += 1
        node.total_visits += 1

        # Incremental mean update of Q
        n = node.n_matrix[a1, a2]
        node.q_matrix[a1, a2] += (value - node.q_matrix[a1, a2]) / n

        return value

    def _select_action_pair(self, node: PUCTNode) -> Tuple[int, int]:
        """Selects simultaneous action pair (a1, a2) balancing Q-value and pUCT exploration."""
        sqrt_total = np.sqrt(max(1, node.total_visits))

        # P1 exploration bonus
        u_p1 = self.c_puct * node.prior_p1 * (sqrt_total / (1.0 + node.n_p1))
        # P2 exploration bonus
        u_p2 = self.c_puct * node.prior_p2 * (sqrt_total / (1.0 + node.n_p2))

        # For P1: maximize Q + U
        q_p1 = np.mean(node.q_matrix, axis=1)
        score_p1 = q_p1 + u_p1
        score_p1[~node.valid_mask_p1] = -1e9
        best_a1 = int(np.argmax(score_p1))

        # For P2: minimize Q - U
        q_p2 = np.mean(node.q_matrix, axis=0)
        score_p2 = q_p2 - u_p2
        score_p2[~node.valid_mask_p2] = 1e9
        best_a2 = int(np.argmin(score_p2))

        return best_a1, best_a2
