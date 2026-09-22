"""
Simultaneous-move pUCT lookahead search with CFR regret matching.
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

        self.q_matrix = np.zeros((9, 9), dtype=np.float32)
        self.n_matrix = np.zeros((9, 9), dtype=np.int32)
        self.n_p1 = np.zeros(9, dtype=np.int32)
        self.n_p2 = np.zeros(9, dtype=np.int32)
        self.total_visits = 0

        self.children: Dict[Tuple[int, int], "PUCTNode"] = {}


class PUCTSearchEngine:
    def __init__(
        self,
        model: ChronosTransformer,
        params: Dict,
        c_puct: float = 1.5,
        max_depth: int = 3,
        default_time_limit_sec: float = 2.0,
    ):
        self.model = model
        self.params = params
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.default_time_limit_sec = default_time_limit_sec

    def evaluate_state(self, state: BattleState) -> Tuple[np.ndarray, np.ndarray, float]:
        inp_p1 = state_to_model_inputs(state, perspective_player=0)
        batched_p1 = {k: v[None, ...] for k, v in inp_p1.items()}
        mask = np.array(get_valid_actions_mask(state))
        mask_p1 = jnp.array(mask[0:1])
        _, probs_p1, val_p1 = self.model.apply(self.params, batched_p1, valid_mask=mask_p1)

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
        start_time = time.perf_counter()
        time_limit = time_limit_sec or self.default_time_limit_sec

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
        while sim_count < max_simulations:
            elapsed = time.perf_counter() - start_time
            if elapsed >= time_limit:
                break

            self._simulate(root, depth=0)
            sim_count += 1

        valid_acts_p1 = root.valid_actions_p1
        valid_acts_p2 = root.valid_actions_p2

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
        if node.is_terminal or depth >= self.max_depth:
            return node.value

        a1, a2 = self._select_action_pair(node)
        pair = (a1, a2)

        if pair not in node.children:
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

        node.n_matrix[a1, a2] += 1
        node.n_p1[a1] += 1
        node.n_p2[a2] += 1
        node.total_visits += 1

        n = node.n_matrix[a1, a2]
        node.q_matrix[a1, a2] += (value - node.q_matrix[a1, a2]) / n

        return value

    def _select_action_pair(self, node: PUCTNode) -> Tuple[int, int]:
        sqrt_total = np.sqrt(max(1, node.total_visits))

        u_p1 = self.c_puct * node.prior_p1 * (sqrt_total / (1.0 + node.n_p1))
        u_p2 = self.c_puct * node.prior_p2 * (sqrt_total / (1.0 + node.n_p2))

        mask_p2 = node.valid_mask_p2.astype(np.float32)
        num_valid_p2 = max(1.0, float(np.sum(mask_p2)))
        q_p1 = np.sum(node.q_matrix * mask_p2[None, :], axis=1) / num_valid_p2
        score_p1 = q_p1 + u_p1
        score_p1[~node.valid_mask_p1] = -1e9
        best_a1 = int(np.argmax(score_p1))

        mask_p1 = node.valid_mask_p1.astype(np.float32)
        num_valid_p1 = max(1.0, float(np.sum(mask_p1)))
        q_p2 = np.sum(node.q_matrix * mask_p1[:, None], axis=0) / num_valid_p1
        score_p2 = q_p2 - u_p2
        score_p2[~node.valid_mask_p2] = 1e9
        best_a2 = int(np.argmin(score_p2))

        return best_a1, best_a2
