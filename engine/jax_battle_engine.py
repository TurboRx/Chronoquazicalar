"""
Gen 9 Random Battles simulator implemented in pure JAX.
"""

from functools import partial
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from engine.battle_state import (
    BattleState,
    STATUS_NONE,
    STATUS_BRN,
    STATUS_PAR,
    STATUS_PSN,
    STATUS_TOX,
    STATUS_SLP,
    STATUS_FRZ,
    WEATHER_NONE,
    WEATHER_SUN,
    WEATHER_RAIN,
    WEATHER_SAND,
    WEATHER_SNOW,
    TERRAIN_NONE,
    TERRAIN_ELECTRIC,
    TERRAIN_GRASSY,
    TERRAIN_MISTY,
    TERRAIN_PSYCHIC,
)

TABLES_PATH = Path(__file__).resolve().parent / "data" / "mechanics_tables.npz"
if not TABLES_PATH.exists():
    raise FileNotFoundError(f"Mechanics tables not found at {TABLES_PATH}")

_tables = np.load(TABLES_PATH)
TYPE_CHART = jnp.array(_tables["type_chart"], dtype=jnp.float32)       # (18, 18)
SPECIES_TABLE = jnp.array(_tables["species_table"], dtype=jnp.int32)    # (N_species, 8)
MOVE_TABLE = jnp.array(_tables["move_table"], dtype=jnp.int32)          # (N_moves, 6)

ROCK_TYPE_IDX = 12
FLYING_TYPE_IDX = 9


def get_stage_multiplier(stage: jnp.ndarray) -> jnp.ndarray:
    s = jnp.clip(stage, -6, 6)
    pos_mult = (2.0 + s) / 2.0
    neg_mult = 2.0 / (2.0 - s)
    return jnp.where(s >= 0, pos_mult, neg_mult)


def calc_raw_damage(
    attacker_stats: jnp.ndarray,
    attacker_boosts: jnp.ndarray,
    defender_stats: jnp.ndarray,
    defender_boosts: jnp.ndarray,
    move_idx: jnp.ndarray,
    attacker_types: jnp.ndarray,
    defender_types: jnp.ndarray,
    attacker_status: jnp.ndarray,
    weather: jnp.ndarray,
    rng_roll: jnp.ndarray,
) -> jnp.ndarray:
    mdata = MOVE_TABLE[move_idx]
    m_type = mdata[0]
    m_cat = mdata[1]  # 0=Phys, 1=Spec, 2=Status
    m_bp = mdata[2]

    is_damage_move = (m_cat != 2) & (m_bp > 0)

    atk_val = jnp.where(
        m_cat == 0,
        attacker_stats[1] * get_stage_multiplier(attacker_boosts[0]),
        attacker_stats[3] * get_stage_multiplier(attacker_boosts[2]),
    )
    atk_val = jnp.where((m_cat == 0) & (attacker_status == STATUS_BRN), atk_val * 0.5, atk_val)

    def_val = jnp.where(
        m_cat == 0,
        defender_stats[2] * get_stage_multiplier(defender_boosts[1]),
        defender_stats[4] * get_stage_multiplier(defender_boosts[3]),
    )
    def_val = jnp.maximum(def_val, 1.0)

    level_factor = 34.0  # floor(2 * 80 / 5 + 2)
    base_dmg = jnp.floor((level_factor * m_bp * atk_val) / (def_val * 50.0) + 2.0)

    sun_mod = jnp.where(m_type == 1, 1.5, jnp.where(m_type == 2, 0.5, 1.0))
    rain_mod = jnp.where(m_type == 2, 1.5, jnp.where(m_type == 1, 0.5, 1.0))
    weather_mod = jnp.where(weather == WEATHER_SUN, sun_mod, jnp.where(weather == WEATHER_RAIN, rain_mod, 1.0))

    is_stab = (m_type == attacker_types[0]) | ((attacker_types[1] >= 0) & (m_type == attacker_types[1]))
    stab_mod = jnp.where(is_stab, 1.5, 1.0)

    eff1 = TYPE_CHART[m_type, defender_types[0]]
    eff2 = jnp.where(defender_types[1] >= 0, TYPE_CHART[m_type, defender_types[1]], 1.0)
    type_eff = eff1 * eff2

    mult = weather_mod * stab_mod * type_eff * rng_roll
    final_dmg = jnp.floor(base_dmg * mult)

    return jnp.where(is_damage_move & (type_eff > 0.0), jnp.maximum(final_dmg, 1.0), 0.0)


def calc_stats_from_base(base_stats: jnp.ndarray, level: int = 80) -> Tuple[jnp.ndarray, jnp.ndarray]:
    hp_stat = jnp.floor(((2.0 * base_stats[0] + 31.0 + 21.0) * level) / 100.0) + level + 10.0
    other_stats = jnp.floor(((2.0 * base_stats[1:] + 31.0 + 21.0) * level) / 100.0) + 5.0
    full_stats = jnp.concatenate([jnp.array([hp_stat]), other_stats])
    return hp_stat, full_stats


def init_battle(rng_key: jax.Array, p1_team: jnp.ndarray, p2_team: jnp.ndarray, p1_moves: jnp.ndarray, p2_moves: jnp.ndarray) -> BattleState:
    act_sp_0 = p1_team[0]
    act_sp_1 = p2_team[0]
    active_species = jnp.array([act_sp_0, act_sp_1], dtype=jnp.int32)

    sp_data_0 = SPECIES_TABLE[act_sp_0]
    sp_data_1 = SPECIES_TABLE[act_sp_1]

    hp_0, stats_0 = calc_stats_from_base(sp_data_0[:6])
    hp_1, stats_1 = calc_stats_from_base(sp_data_1[:6])

    active_max_hp = jnp.array([hp_0, hp_1], dtype=jnp.float32)
    active_current_hp = jnp.array([hp_0, hp_1], dtype=jnp.float32)
    active_hp = jnp.array([1.0, 1.0], dtype=jnp.float32)
    active_stats = jnp.stack([stats_0, stats_1])
    active_boosts = jnp.zeros((2, 7), dtype=jnp.int32)
    active_status = jnp.zeros((2,), dtype=jnp.int32)

    active_moves = jnp.stack([p1_moves[0], p2_moves[0]])
    active_move_pp = jnp.ones((2, 4), dtype=jnp.float32)
    active_types = jnp.stack([sp_data_0[6:8], sp_data_1[6:8]])

    team_species = jnp.stack([p1_team, p2_team])
    team_hp = jnp.ones((2, 6), dtype=jnp.float32)
    team_alive = jnp.ones((2, 6), dtype=jnp.bool_)

    hazards = jnp.zeros((2, 4), dtype=jnp.int32)

    return BattleState(
        active_species=active_species,
        active_hp=active_hp,
        active_max_hp=active_max_hp,
        active_current_hp=active_current_hp,
        active_stats=active_stats,
        active_boosts=active_boosts,
        active_status=active_status,
        active_moves=active_moves,
        active_move_pp=active_move_pp,
        active_types=active_types,
        team_species=team_species,
        team_hp=team_hp,
        team_alive=team_alive,
        weather=jnp.array(WEATHER_NONE, dtype=jnp.int32),
        weather_turns=jnp.array(0, dtype=jnp.int32),
        terrain=jnp.array(TERRAIN_NONE, dtype=jnp.int32),
        terrain_turns=jnp.array(0, dtype=jnp.int32),
        hazards=hazards,
        turn_count=jnp.array(0, dtype=jnp.int32),
        rng_key=rng_key,
        done=jnp.array(False, dtype=jnp.bool_),
        winner=jnp.array(0, dtype=jnp.int32),
    )


def get_valid_actions_mask(state: BattleState) -> jnp.ndarray:
    move_valid = (state.active_moves > 0) & (state.active_move_pp > 0) & (state.active_hp[:, None] > 0)
    switch_valid = state.team_alive[:, 1:6]
    return jnp.concatenate([move_valid, switch_valid], axis=-1)


def execute_switch(
    state: BattleState,
    player_idx: int,
    bench_slot: int,
) -> BattleState:
    opp_idx = 1 - player_idx

    old_act_sp = state.active_species[player_idx]
    old_act_hp = state.active_hp[player_idx]
    old_act_alive = old_act_hp > 0

    new_act_sp = state.team_species[player_idx, bench_slot]
    sp_data = SPECIES_TABLE[new_act_sp]
    new_max_hp, new_stats = calc_stats_from_base(sp_data[:6])
    new_types = sp_data[6:8]

    sr_active = state.hazards[player_idx, 0] > 0
    sr_eff = TYPE_CHART[ROCK_TYPE_IDX, new_types[0]] * jnp.where(
        new_types[1] >= 0, TYPE_CHART[ROCK_TYPE_IDX, new_types[1]], 1.0
    )
    sr_dmg = jnp.where(sr_active, jnp.floor(new_max_hp * 0.125 * sr_eff), 0.0)

    spikes_layers = state.hazards[player_idx, 1]
    is_flying = (new_types[0] == FLYING_TYPE_IDX) | (new_types[1] == FLYING_TYPE_IDX)
    spikes_frac = jnp.where(spikes_layers == 1, 0.125, jnp.where(spikes_layers == 2, 0.1667, jnp.where(spikes_layers >= 3, 0.25, 0.0)))
    spikes_dmg = jnp.where((spikes_layers > 0) & ~is_flying, jnp.floor(new_max_hp * spikes_frac), 0.0)

    total_hazard_dmg = sr_dmg + spikes_dmg
    incoming_current_hp = jnp.maximum(new_max_hp * state.team_hp[player_idx, bench_slot] - total_hazard_dmg, 0.0)
    incoming_hp_frac = incoming_current_hp / new_max_hp
    incoming_alive = incoming_hp_frac > 0

    web_active = (state.hazards[player_idx, 3] > 0) & ~is_flying
    new_boosts = jnp.zeros(7, dtype=jnp.int32)
    new_boosts = new_boosts.at[4].set(jnp.where(web_active, -1, 0))

    act_sp = state.active_species.at[player_idx].set(new_act_sp)
    act_hp = state.active_hp.at[player_idx].set(incoming_hp_frac)
    act_max_hp = state.active_max_hp.at[player_idx].set(new_max_hp)
    act_curr_hp = state.active_current_hp.at[player_idx].set(incoming_current_hp)
    act_stats = state.active_stats.at[player_idx].set(new_stats)
    act_boosts = state.active_boosts.at[player_idx].set(new_boosts)
    act_types = state.active_types.at[player_idx].set(new_types)

    team_sp = state.team_species.at[player_idx, 0].set(new_act_sp).at[player_idx, bench_slot].set(old_act_sp)
    team_hp = state.team_hp.at[player_idx, 0].set(incoming_hp_frac).at[player_idx, bench_slot].set(old_act_hp)
    team_al = state.team_alive.at[player_idx, 0].set(incoming_alive).at[player_idx, bench_slot].set(old_act_alive)

    return state.replace(
        active_species=act_sp,
        active_hp=act_hp,
        active_max_hp=act_max_hp,
        active_current_hp=act_curr_hp,
        active_stats=act_stats,
        active_boosts=act_boosts,
        active_types=act_types,
        team_species=team_sp,
        team_hp=team_hp,
        team_alive=team_al,
    )


def execute_attack(
    state: BattleState,
    attacker_idx: int,
    move_slot: int,
    rng_roll: jnp.ndarray,
) -> BattleState:
    defender_idx = 1 - attacker_idx
    move_idx = state.active_moves[attacker_idx, move_slot]

    dmg = calc_raw_damage(
        attacker_stats=state.active_stats[attacker_idx],
        attacker_boosts=state.active_boosts[attacker_idx],
        defender_stats=state.active_stats[defender_idx],
        defender_boosts=state.active_boosts[defender_idx],
        move_idx=move_idx,
        attacker_types=state.active_types[attacker_idx],
        defender_types=state.active_types[defender_idx],
        attacker_status=state.active_status[attacker_idx],
        weather=state.weather,
        rng_roll=rng_roll,
    )

    new_curr_hp = jnp.maximum(state.active_current_hp[defender_idx] - dmg, 0.0)
    new_hp_frac = new_curr_hp / state.active_max_hp[defender_idx]
    is_alive = new_hp_frac > 0

    pp = state.active_move_pp.at[attacker_idx, move_slot].set(
        jnp.maximum(state.active_move_pp[attacker_idx, move_slot] - 0.05, 0.0)
    )

    act_curr = state.active_current_hp.at[defender_idx].set(new_curr_hp)
    act_hp = state.active_hp.at[defender_idx].set(new_hp_frac)
    team_hp = state.team_hp.at[defender_idx, 0].set(new_hp_frac)
    team_al = state.team_alive.at[defender_idx, 0].set(is_alive)

    return state.replace(
        active_current_hp=act_curr,
        active_hp=act_hp,
        active_move_pp=pp,
        team_hp=team_hp,
        team_alive=team_al,
    )


def auto_faint_switch(state: BattleState, player_idx: int) -> BattleState:
    needs_switch = state.active_hp[player_idx] <= 0
    alive_bench = state.team_alive[player_idx, 1:6]
    has_alive = jnp.any(alive_bench)
    first_alive_idx = jnp.argmax(alive_bench) + 1

    def do_switch(s: BattleState) -> BattleState:
        return execute_switch(s, player_idx, first_alive_idx)

    return jax.lax.cond(needs_switch & has_alive, do_switch, lambda s: s, state)


@jax.jit
def step(
    state: BattleState,
    action_p1: jnp.ndarray,
    action_p2: jnp.ndarray,
) -> Tuple[BattleState, jnp.ndarray, jnp.ndarray]:
    key, k_roll1, k_roll2, k_tie = jax.random.split(state.rng_key, 4)
    rng_roll1 = jax.random.uniform(k_roll1, shape=(), minval=0.85, maxval=1.0)
    rng_roll2 = jax.random.uniform(k_roll2, shape=(), minval=0.85, maxval=1.0)
    tie_breaker = jax.random.bernoulli(k_tie, p=0.5)

    pri_0 = jnp.where(action_p1 >= 4, 6, MOVE_TABLE[state.active_moves[0, jnp.clip(action_p1, 0, 3)], 4])
    pri_1 = jnp.where(action_p2 >= 4, 6, MOVE_TABLE[state.active_moves[1, jnp.clip(action_p2, 0, 3)], 4])

    spe_0 = state.active_stats[0, 5] * get_stage_multiplier(state.active_boosts[0, 4]) * jnp.where(state.active_status[0] == STATUS_PAR, 0.5, 1.0)
    spe_1 = state.active_stats[1, 5] * get_stage_multiplier(state.active_boosts[1, 4]) * jnp.where(state.active_status[1] == STATUS_PAR, 0.5, 1.0)

    p1_first = (pri_0 > pri_1) | ((pri_0 == pri_1) & (spe_0 > spe_1)) | ((pri_0 == pri_1) & (spe_0 == spe_1) & tie_breaker)

    first_p = jnp.where(p1_first, 0, 1)
    second_p = 1 - first_p
    first_act = jnp.where(p1_first, action_p1, action_p2)
    second_act = jnp.where(p1_first, action_p2, action_p1)
    first_roll = jnp.where(p1_first, rng_roll1, rng_roll2)
    second_roll = jnp.where(p1_first, rng_roll2, rng_roll1)

    def execute_actor(s: BattleState, act: jnp.ndarray, p_idx: int, roll: jnp.ndarray) -> BattleState:
        is_switch = act >= 4
        bench_slot = act - 3
        move_slot = jnp.clip(act, 0, 3)

        def do_sw(st: BattleState) -> BattleState:
            return execute_switch(st, p_idx, bench_slot)

        def do_mv(st: BattleState) -> BattleState:
            return execute_attack(st, p_idx, move_slot, roll)

        return jax.lax.cond(is_switch, do_sw, do_mv, s)

    s1 = execute_actor(state, first_act, first_p, first_roll)
    second_alive = s1.active_hp[second_p] > 0

    def resolve_second(st: BattleState) -> BattleState:
        return execute_actor(st, second_act, second_p, second_roll)

    s2 = jax.lax.cond(second_alive, resolve_second, lambda st: st, s1)

    s3 = auto_faint_switch(s2, 0)
    s4 = auto_faint_switch(s3, 1)

    def apply_status_dmg(st: BattleState, p: int) -> BattleState:
        st_val = st.active_status[p]
        dmg_frac = jnp.where(st_val == STATUS_BRN, 1.0 / 16.0, jnp.where(st_val == STATUS_PSN, 1.0 / 8.0, 0.0))
        dmg = jnp.floor(st.active_max_hp[p] * dmg_frac)
        curr = jnp.maximum(st.active_current_hp[p] - dmg, 0.0)
        hp = curr / st.active_max_hp[p]
        al = hp > 0
        return st.replace(
            active_current_hp=st.active_current_hp.at[p].set(curr),
            active_hp=st.active_hp.at[p].set(hp),
            team_hp=st.team_hp.at[p, 0].set(hp),
            team_alive=st.team_alive.at[p, 0].set(al),
        )

    s5 = apply_status_dmg(s4, 0)
    s6 = apply_status_dmg(s5, 1)

    s7 = auto_faint_switch(s6, 0)
    s8 = auto_faint_switch(s7, 1)

    p1_has_alive = jnp.any(s8.team_alive[0])
    p2_has_alive = jnp.any(s8.team_alive[1])
    turn_limit_reached = s8.turn_count >= 100

    done = (~p1_has_alive) | (~p2_has_alive) | turn_limit_reached
    winner = jnp.where(p1_has_alive & ~p2_has_alive, 1, jnp.where(p2_has_alive & ~p1_has_alive, 2, 0))

    p1_team_hp = jnp.mean(s8.team_hp[0])
    p2_team_hp = jnp.mean(s8.team_hp[1])
    hp_reward = (p1_team_hp - p2_team_hp) * 0.1
    win_reward = jnp.where(winner == 1, 1.0, jnp.where(winner == 2, -1.0, 0.0))
    reward = jnp.where(done, win_reward, hp_reward)

    next_state = s8.replace(
        turn_count=s8.turn_count + 1,
        rng_key=key,
        done=done,
        winner=winner,
    )

    return next_state, reward, done


batch_step = jax.jit(jax.vmap(step, in_axes=(0, 0, 0)))


def _scan_step(carry, _):
    s, a1, a2 = carry
    ns, r, d = batch_step(s, a1, a2)
    return (ns, a1, a2), (r, d)


@partial(jax.jit, static_argnums=(3,))
def batched_rollout(
    initial_states: BattleState,
    actions_p1: jnp.ndarray,
    actions_p2: jnp.ndarray,
    num_steps: int,
) -> Tuple[BattleState, jnp.ndarray, jnp.ndarray]:
    (final_states, _, _), (rewards, dones) = jax.lax.scan(
        _scan_step, (initial_states, actions_p1, actions_p2), None, length=num_steps
    )
    return final_states, rewards, dones
