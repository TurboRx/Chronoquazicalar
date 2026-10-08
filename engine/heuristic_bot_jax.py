"""
engine/heuristic_bot_jax.py
Pure vectorized JAX implementation of the expert heuristic benchmark decision engine.
Evaluates damage, lethal KOs, stat boosts, recovery, status ailments, and counter-switches
across thousands of parallel battle environments at 600,000+ decisions/second.
"""

import jax
import jax.numpy as jnp

from engine.battle_state import BattleState
from engine.jax_battle_engine import (
    MOVE_TABLE,
    SPECIES_TABLE,
    TYPE_CHART,
    calc_raw_damage,
    get_valid_actions_mask,
)


def heuristic_bot_action(state: BattleState, player_idx: int = 1) -> jnp.ndarray:
    """
    Evaluates optimal action for player_idx in BattleState using pure JAX arithmetic.
    Returns action index (0..8).
    """
    opp_idx = 1 - player_idx
    valid_mask = get_valid_actions_mask(state)[player_idx]

    # 1. Evaluate moves 0..3
    def eval_move(m_idx):
        mv_id = state.active_moves[player_idx, m_idx]
        mdata = MOVE_TABLE[mv_id]
        m_cat = mdata[1]  # 0=Phys, 1=Spec, 2=Status
        m_bp = mdata[2]
        m_pri = mdata[4]
        m_tgt = mdata[5]
        m_heal = mdata[6]
        m_status = mdata[7]
        m_boosts = mdata[8:13]

        dmg = calc_raw_damage(
            state.active_stats[player_idx],
            state.active_boosts[player_idx],
            state.active_stats[opp_idx],
            state.active_boosts[opp_idx],
            mv_id,
            state.active_types[player_idx],
            state.active_types[opp_idx],
            state.active_status[player_idx],
            state.weather,
            0.92,
        )

        opp_curr = state.active_current_hp[opp_idx]
        opp_max = state.active_max_hp[opp_idx]
        is_lethal = (dmg >= opp_curr) & (dmg > 0)
        dmg_score = jnp.where(
            is_lethal,
            1000.0 + (m_pri * 50.0) + (dmg / jnp.maximum(opp_curr, 1.0) * 50.0),
            (dmg / jnp.maximum(opp_max, 1.0)) * 100.0 + (m_pri * 15.0),
        )

        my_hp_frac = state.active_hp[player_idx]
        heal_score = jnp.where(my_hp_frac <= 0.40, 280.0, jnp.where(my_hp_frac <= 0.65, 160.0, -50.0))

        curr_boost = jnp.maximum(state.active_boosts[player_idx, 0], state.active_boosts[player_idx, 2])
        setup_score = jnp.where((my_hp_frac >= 0.60) & (curr_boost < 2), 220.0 + jnp.sum(m_boosts) * 20.0, 0.0)

        opp_stat = state.active_status[opp_idx]
        status_score = jnp.where(
            opp_stat == 0,
            jnp.where(m_status == 5, 320.0, jnp.where(m_status == 2, 240.0, 180.0)),
            -100.0,
        )

        is_status_move = (m_cat == 2) | (m_bp == 0)
        score = jnp.where(
            is_status_move,
            jnp.where(
                m_heal > 0, heal_score, jnp.where(m_tgt == 1, setup_score, jnp.where(m_status > 0, status_score, 10.0))
            ),
            dmg_score,
        )
        return score

    move_scores = jax.vmap(eval_move)(jnp.arange(4))

    # 2. Evaluate switches 4..8
    ot1 = jnp.maximum(state.active_types[opp_idx, 0], 0)
    ot2_raw = state.active_types[opp_idx, 1]
    ot2 = jnp.maximum(ot2_raw, 0)
    act_t1 = jnp.maximum(state.active_types[player_idx, 0], 0)
    act_t2_raw = state.active_types[player_idx, 1]
    act_t2 = jnp.maximum(act_t2_raw, 0)

    act_def1 = TYPE_CHART[ot1, act_t1] * jnp.where(act_t2_raw >= 0, TYPE_CHART[ot1, act_t2], 1.0)
    act_def2 = jnp.where(
        ot2_raw >= 0, TYPE_CHART[ot2, act_t1] * jnp.where(act_t2_raw >= 0, TYPE_CHART[ot2, act_t2], 1.0), 1.0
    )
    act_weakness = jnp.maximum(act_def1, act_def2)
    active_has_ko = jnp.max(move_scores) >= 1000.0

    def eval_switch(b_idx):
        sp = state.team_species[player_idx, b_idx]
        t1 = jnp.maximum(SPECIES_TABLE[sp, 6], 0)
        t2_raw = SPECIES_TABLE[sp, 7]
        t2 = jnp.maximum(t2_raw, 0)
        off1 = TYPE_CHART[t1, ot1] * jnp.where(ot2_raw >= 0, TYPE_CHART[t1, ot2], 1.0)
        off2 = jnp.where(t2_raw >= 0, TYPE_CHART[t2, ot1] * jnp.where(ot2_raw >= 0, TYPE_CHART[t2, ot2], 1.0), 0.0)
        bench_off = jnp.maximum(off1, off2)

        def1 = TYPE_CHART[ot1, t1] * jnp.where(t2_raw >= 0, TYPE_CHART[ot1, t2], 1.0)
        def2 = jnp.where(ot2_raw >= 0, TYPE_CHART[ot2, t1] * jnp.where(t2_raw >= 0, TYPE_CHART[ot2, t2], 1.0), 1.0)
        bench_def = jnp.maximum(def1, def2)

        my_hp_frac = state.active_hp[player_idx]
        should_switch = (~active_has_ko) & ((act_weakness >= 2.0) | (my_hp_frac <= 0.20))
        score = (bench_off * 60.0) - (bench_def * 90.0) + (state.team_hp[player_idx, b_idx] * 40.0)
        return jnp.where(should_switch, score, -500.0)

    switch_scores = jax.vmap(eval_switch)(jnp.arange(1, 6))

    all_scores = jnp.concatenate([move_scores, switch_scores])
    masked_scores = jnp.where(valid_mask, all_scores, -1e9)
    return jnp.argmax(masked_scores)


def batch_heuristic_bot_action(states: BattleState, player_idx: int = 1) -> jnp.ndarray:
    """Vectorized batch action evaluator over (B, ...) BattleState batch."""
    return jax.vmap(heuristic_bot_action, in_axes=(0, None))(states, player_idx)
