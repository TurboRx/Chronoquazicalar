"""
Counterfactual Regret Matching (CFR) solver for two-player simultaneous games.
"""

from typing import Tuple

import numpy as np


def solve_matrix_game_cfr(
    payoff_matrix: np.ndarray,
    num_iterations: int = 500,
    regret_floor: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes approximate Nash equilibrium mixed strategies for zero-sum matrix games.

    Args:
        payoff_matrix: (m, n) utility matrix for Player 1.
        num_iterations: Number of CFR iterations.
        regret_floor: Minimum regret floor.

    Returns:
        p1_strategy, p2_strategy, game_value
    """
    m, n = payoff_matrix.shape
    if m == 0 or n == 0:
        return np.zeros(m, dtype=np.float64), np.zeros(n, dtype=np.float64), 0.0

    regrets_p1 = np.zeros(m, dtype=np.float64)
    regrets_p2 = np.zeros(n, dtype=np.float64)

    strategy_sum_p1 = np.zeros(m, dtype=np.float64)
    strategy_sum_p2 = np.zeros(n, dtype=np.float64)

    for _ in range(num_iterations):
        pos_regrets_p1 = np.maximum(regrets_p1, regret_floor)
        sum_pos_p1 = np.sum(pos_regrets_p1)
        sigma_p1 = (pos_regrets_p1 / sum_pos_p1) if sum_pos_p1 > 1e-12 else np.full(m, 1.0 / m, dtype=np.float64)

        pos_regrets_p2 = np.maximum(regrets_p2, regret_floor)
        sum_pos_p2 = np.sum(pos_regrets_p2)
        sigma_p2 = (pos_regrets_p2 / sum_pos_p2) if sum_pos_p2 > 1e-12 else np.full(n, 1.0 / n, dtype=np.float64)

        strategy_sum_p1 += sigma_p1
        strategy_sum_p2 += sigma_p2

        u1_action = np.dot(payoff_matrix, sigma_p2)
        u1_expected = np.dot(sigma_p1, u1_action)

        u2_action = -np.dot(sigma_p1, payoff_matrix)
        u2_expected = -u1_expected

        regrets_p1 += u1_action - u1_expected
        regrets_p2 += u2_action - u2_expected

    sum_s1 = np.sum(strategy_sum_p1)
    p1_strategy = (strategy_sum_p1 / sum_s1) if sum_s1 > 1e-12 else np.full(m, 1.0 / m)

    sum_s2 = np.sum(strategy_sum_p2)
    p2_strategy = (strategy_sum_p2 / sum_s2) if sum_s2 > 1e-12 else np.full(n, 1.0 / n)

    game_value = float(np.dot(p1_strategy, np.dot(payoff_matrix, p2_strategy)))

    return p1_strategy, p2_strategy, game_value


def get_mixed_action(
    valid_actions: np.ndarray,
    payoff_matrix: np.ndarray,
    num_iterations: int = 500,
    temperature: float = 1.0,
) -> Tuple[int, np.ndarray]:
    if len(valid_actions) == 0:
        return 0, np.zeros(9, dtype=np.float32)

    p1_strat, _, _ = solve_matrix_game_cfr(payoff_matrix, num_iterations=num_iterations)

    sum_p = np.sum(p1_strat)
    if sum_p > 1e-12:
        norm_p = p1_strat / sum_p
    else:
        norm_p = np.full(len(valid_actions), 1.0 / len(valid_actions), dtype=np.float64)

    full_strategy = np.zeros(9, dtype=np.float32)
    for idx, act in enumerate(valid_actions):
        full_strategy[act] = float(norm_p[idx])

    if temperature <= 1e-4:
        best_local_idx = int(np.argmax(norm_p))
        chosen_action = int(valid_actions[best_local_idx])
    else:
        scaled_logits = np.log(np.maximum(norm_p, 1e-12)) / max(temperature, 1e-4)
        scaled_logits -= np.max(scaled_logits)
        exp_p = np.exp(scaled_logits)
        sampling_p = exp_p / np.sum(exp_p)
        chosen_action = int(np.random.choice(valid_actions, p=sampling_p))

    return chosen_action, full_strategy
