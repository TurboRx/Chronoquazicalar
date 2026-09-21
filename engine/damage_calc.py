"""
Damage calculation and deterministic KO heuristics for search pruning.
"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp

from engine.battle_state import BattleState
from engine.jax_battle_engine import (
    calc_raw_damage,
    get_stage_multiplier,
    MOVE_TABLE,
    SPECIES_TABLE,
    TYPE_CHART,
    STATUS_PAR,
)


@jax.jit
def get_min_roll_damages(state: BattleState, player_idx: int = 0) -> jnp.ndarray:
    opp_idx = 1 - player_idx

    def eval_move(move_idx: jnp.ndarray) -> jnp.ndarray:
        return calc_raw_damage(
            attacker_stats=state.active_stats[player_idx],
            attacker_boosts=state.active_boosts[player_idx],
            defender_stats=state.active_stats[opp_idx],
            defender_boosts=state.active_boosts[opp_idx],
            move_idx=move_idx,
            attacker_types=state.active_types[player_idx],
            defender_types=state.active_types[opp_idx],
            attacker_status=state.active_status[player_idx],
            weather=state.weather,
            rng_roll=jnp.array(0.85, dtype=jnp.float32),
        )

    moves = state.active_moves[player_idx]
    return jax.vmap(eval_move)(moves)


@jax.jit
def check_guaranteed_ko(state: BattleState, player_idx: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns (has_ko, best_action).
    has_ko is True if any valid move guarantees KO under minimum damage roll and outspeeds.
    """
    opp_idx = 1 - player_idx
    opp_curr_hp = state.active_current_hp[opp_idx]

    min_damages = get_min_roll_damages(state, player_idx)

    pp_valid = state.active_move_pp[player_idx] > 0
    move_valid = (state.active_moves[player_idx] > 0) & pp_valid
    is_ko = (min_damages >= opp_curr_hp) & move_valid & (min_damages > 0)

    my_spe = state.active_stats[player_idx, 5] * get_stage_multiplier(state.active_boosts[player_idx, 4])
    my_spe = jnp.where(state.active_status[player_idx] == STATUS_PAR, my_spe * 0.5, my_spe)

    opp_spe = state.active_stats[opp_idx, 5] * get_stage_multiplier(state.active_boosts[opp_idx, 4])
    opp_spe = jnp.where(state.active_status[opp_idx] == STATUS_PAR, opp_spe * 0.5, opp_spe)

    outspeeds = my_spe > opp_spe

    move_ids = state.active_moves[player_idx]
    priorities = MOVE_TABLE[move_ids, 4]

    moves_first = (priorities > 0) | ((priorities >= 0) & outspeeds)
    guaranteed_mask = is_ko & moves_first

    has_ko = jnp.any(guaranteed_mask)
    score = jnp.where(guaranteed_mask, priorities * 10000.0 + min_damages, -1.0)
    best_action = jnp.where(has_ko, jnp.argmax(score), -1)

    return has_ko, best_action
