"""
search/regret_matching.py
Phase 6: Counterfactual Regret Matching (CFR) and Nash Equilibrium solver
for simultaneous turn matrix games in Pokémon Showdown.
Prevents the bot from being predictable on 50/50 prediction turns.
"""

from typing import Tuple
import numpy as np


def solve_matrix_game_cfr(
    payoff_matrix: np.ndarray,
    num_iterations: int = 500,
    regret_floor: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes approximate Nash equilibrium mixed strategies for a two-player zero-sum
    simultaneous move game using Counterfactual Regret Matching (CFR).

    Args:
        payoff_matrix: (m, n) float array.
                       payoff_matrix[i, j] is the utility to Player 1 (Player 2 utility is -utility).
        num_iterations: Number of CFR iterations to run.
        regret_floor: Minimum regret value (0.0 for standard CFR+ / regret matching).

    Returns:
        p1_strategy: (m,) probability distribution over Player 1 actions.
        p2_strategy: (n,) probability distribution over Player 2 actions.
        game_value: Expected payoff for Player 1 under the equilibrium strategies.
    """
    m, n = payoff_matrix.shape

    # Cumulative regrets
    regrets_p1 = np.zeros(m, dtype=np.float64)
    regrets_p2 = np.zeros(n, dtype=np.float64)

    # Strategy accumulators for average strategy
    strategy_sum_p1 = np.zeros(m, dtype=np.float64)
    strategy_sum_p2 = np.zeros(n, dtype=np.float64)

    for _ in range(num_iterations):
        # 1. Compute current strategies from positive regrets
        pos_regrets_p1 = np.maximum(regrets_p1, regret_floor)
        sum_pos_p1 = np.sum(pos_regrets_p1)
        if sum_pos_p1 > 1e-12:
            sigma_p1 = pos_regrets_p1 / sum_pos_p1
        else:
            sigma_p1 = np.full(m, 1.0 / m, dtype=np.float64)

        pos_regrets_p2 = np.maximum(regrets_p2, regret_floor)
        sum_pos_p2 = np.sum(pos_regrets_p2)
        if sum_pos_p2 > 1e-12:
            sigma_p2 = pos_regrets_p2 / sum_pos_p2
        else:
            sigma_p2 = np.full(n, 1.0 / n, dtype=np.float64)

        # Accumulate strategies
        strategy_sum_p1 += sigma_p1
        strategy_sum_p2 += sigma_p2

        # 2. Compute expected utilities
        # Payoff to P1 for action i against P2 strategy
        u1_action = np.dot(payoff_matrix, sigma_p2)  # (m,)
        u1_expected = np.dot(sigma_p1, u1_action)     # scalar

        # Payoff to P2 for action j against P1 strategy (zero-sum: -P1 payoff)
        u2_action = -np.dot(sigma_p1, payoff_matrix)  # (n,)
        u2_expected = -u1_expected                   # scalar

        # 3. Update cumulative regrets
        regrets_p1 += (u1_action - u1_expected)
        regrets_p2 += (u2_action - u2_expected)

    # Compute average strategies (Nash Equilibrium)
    sum_s1 = np.sum(strategy_sum_p1)
    p1_strategy = (strategy_sum_p1 / sum_s1) if sum_s1 > 1e-12 else np.full(m, 1.0 / m)

    sum_s2 = np.sum(strategy_sum_p2)
    p2_strategy = (strategy_sum_p2 / sum_s2) if sum_s2 > 1e-12 else np.full(n, 1.0 / n)

    # Expected game value under equilibrium
    game_value = float(np.dot(p1_strategy, np.dot(payoff_matrix, p2_strategy)))

    return p1_strategy, p2_strategy, game_value


def get_mixed_action(
    valid_actions: np.ndarray,
    payoff_matrix: np.ndarray,
    num_iterations: int = 500,
    temperature: float = 1.0,
) -> Tuple[int, np.ndarray]:
    """
    Computes the Nash equilibrium strategy over valid actions and samples an action.

    Args:
        valid_actions: 1D array of valid action indices.
        payoff_matrix: (len(valid_actions), len(opp_actions)) utility matrix.
        num_iterations: CFR iterations.
        temperature: Sampling temperature (1.0 for true Nash, 0.0 for greedy argmax).

    Returns:
        chosen_action: Selected action index.
        strategy: Full probability vector over all 9 actions.
    """
    p1_strat, _, _ = solve_matrix_game_cfr(payoff_matrix, num_iterations=num_iterations)

    full_strategy = np.zeros(9, dtype=np.float32)
    for idx, act in enumerate(valid_actions):
        full_strategy[act] = float(p1_strat[idx])

    if temperature <= 1e-4:
        best_local_idx = int(np.argmax(p1_strat))
        chosen_action = int(valid_actions[best_local_idx])
    else:
        chosen_action = int(np.random.choice(valid_actions, p=p1_strat))

    return chosen_action, full_strategy
