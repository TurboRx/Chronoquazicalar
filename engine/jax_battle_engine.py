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
    STATUS_BRN,
    STATUS_NONE,
    STATUS_PAR,
    STATUS_PSN,
    TERRAIN_NONE,
    WEATHER_NONE,
    WEATHER_RAIN,
    WEATHER_SUN,
    BattleState,
)

TABLES_PATH = Path(__file__).resolve().parent / "data" / "mechanics_tables.npz"
if not TABLES_PATH.exists():
    raise FileNotFoundError(f"Mechanics tables not found at {TABLES_PATH}")

_tables = np.load(TABLES_PATH)
TYPE_CHART = jnp.array(_tables["type_chart"], dtype=jnp.float32)  # (18, 18)
SPECIES_TABLE = jnp.array(_tables["species_table"], dtype=jnp.int32)  # (N_species, 8)
MOVE_TABLE = jnp.array(_tables["move_table"], dtype=jnp.int32)  # (N_moves, 6)

PROFILES_PATH = Path(__file__).resolve().parent / "data" / "randbats_profiles.npz"
if PROFILES_PATH.exists():
    _profiles = np.load(PROFILES_PATH)
    RANDBATS_SPECIES = jnp.array(_profiles["species"], dtype=jnp.int32)
    RANDBATS_MOVES = jnp.array(_profiles["moves"], dtype=jnp.int32)
else:
    RANDBATS_SPECIES = jnp.array([950, 230, 150, 1380, 450, 1200], dtype=jnp.int32)
    RANDBATS_MOVES = jnp.array([[100, 200, 300, 400]] * 6, dtype=jnp.int32)
NUM_PROFILES = int(RANDBATS_SPECIES.shape[0])

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
    weather_mod = jnp.where(
        weather == WEATHER_SUN,
        sun_mod,
        jnp.where(weather == WEATHER_RAIN, rain_mod, 1.0),
    )

    is_stab = (m_type == attacker_types[0]) | ((attacker_types[1] >= 0) & (m_type == attacker_types[1]))
    stab_mod = jnp.where(is_stab, 1.5, 1.0)

    eff1 = TYPE_CHART[m_type, defender_types[0]]
    eff2 = jnp.where(defender_types[1] >= 0, TYPE_CHART[m_type, defender_types[1]], 1.0)
    type_eff = eff1 * eff2

    mult = weather_mod * stab_mod * type_eff * rng_roll
    final_dmg = jnp.floor(base_dmg * mult)

    return jnp.where(is_damage_move & (type_eff > 0.0), jnp.maximum(final_dmg, 1.0), 0.0)


def calc_stats_from_base(
    base_stats: jnp.ndarray, level: jnp.ndarray = jnp.array(80, dtype=jnp.int32)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    hp_stat = jnp.floor(((2.0 * base_stats[0] + 31.0 + 21.0) * level) / 100.0) + level + 10.0
    other_stats = jnp.floor(((2.0 * base_stats[1:] + 31.0 + 21.0) * level) / 100.0) + 5.0
    full_stats = jnp.concatenate([jnp.array([hp_stat]), other_stats])
    return hp_stat, full_stats


def sample_random_level(rng_key: jax.Array) -> jnp.ndarray:
    return jax.random.randint(rng_key, (), minval=75, maxval=96)


def init_battle(
    rng_key: jax.Array,
    p1_team: jnp.ndarray,
    p2_team: jnp.ndarray,
    p1_moves: jnp.ndarray,
    p2_moves: jnp.ndarray,
) -> BattleState:
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
    team_moves = jnp.stack([p1_moves, p2_moves])
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
        team_moves=team_moves,
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
    spikes_frac = jnp.where(
        spikes_layers == 1,
        0.125,
        jnp.where(spikes_layers == 2, 0.1667, jnp.where(spikes_layers >= 3, 0.25, 0.0)),
    )
    spikes_dmg = jnp.where((spikes_layers > 0) & ~is_flying, jnp.floor(new_max_hp * spikes_frac), 0.0)

    total_hazard_dmg = sr_dmg + spikes_dmg
    incoming_current_hp = jnp.maximum(new_max_hp * state.team_hp[player_idx, bench_slot] - total_hazard_dmg, 0.0)
    incoming_hp_frac = incoming_current_hp / jnp.maximum(new_max_hp, 1.0)
    incoming_alive = incoming_hp_frac > 0

    web_active = (state.hazards[player_idx, 3] > 0) & ~is_flying
    new_boosts = jnp.zeros(7, dtype=jnp.int32)
    new_boosts = new_boosts.at[4].set(jnp.where(web_active, -1, 0))

    new_act_mvs = state.team_moves[player_idx, bench_slot]
    old_act_mvs = state.active_moves[player_idx]

    act_sp = state.active_species.at[player_idx].set(new_act_sp)
    act_hp = state.active_hp.at[player_idx].set(incoming_hp_frac)
    act_max_hp = state.active_max_hp.at[player_idx].set(new_max_hp)
    act_curr_hp = state.active_current_hp.at[player_idx].set(incoming_current_hp)
    act_stats = state.active_stats.at[player_idx].set(new_stats)
    act_boosts = state.active_boosts.at[player_idx].set(new_boosts)
    act_types = state.active_types.at[player_idx].set(new_types)
    act_mvs = state.active_moves.at[player_idx].set(new_act_mvs)
    act_pp = state.active_move_pp.at[player_idx].set(jnp.ones(4, dtype=jnp.float32))

    team_sp = state.team_species.at[player_idx, 0].set(new_act_sp).at[player_idx, bench_slot].set(old_act_sp)
    team_mvs = state.team_moves.at[player_idx, 0].set(new_act_mvs).at[player_idx, bench_slot].set(old_act_mvs)
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
        active_moves=act_mvs,
        active_move_pp=act_pp,
        team_species=team_sp,
        team_moves=team_mvs,
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
    mdata = MOVE_TABLE[move_idx]
    # mdata: [mtype, cat, bp, acc, pri, target, heal_pct, status_id, b_atk, b_def, b_spa, b_spd, b_spe]
    target = mdata[5]
    heal_pct = mdata[6]
    status_id = mdata[7]
    boosts_5 = mdata[8:13]

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

    new_curr_hp_def = jnp.maximum(state.active_current_hp[defender_idx] - dmg, 0.0)
    new_hp_frac_def = new_curr_hp_def / state.active_max_hp[defender_idx]
    def_alive = new_hp_frac_def > 0

    # Healing application (Recover, Roost, Soft-Boiled, Synthesis, etc.)
    heal_amt = jnp.where(
        heal_pct > 0,
        jnp.floor(state.active_max_hp[attacker_idx] * (heal_pct / 100.0)),
        0.0,
    )
    new_curr_hp_atk = jnp.minimum(
        state.active_current_hp[attacker_idx] + heal_amt,
        state.active_max_hp[attacker_idx],
    )
    new_hp_frac_atk = new_curr_hp_atk / state.active_max_hp[attacker_idx]

    pp = state.active_move_pp.at[attacker_idx, move_slot].set(
        jnp.maximum(state.active_move_pp[attacker_idx, move_slot] - 0.05, 0.0)
    )

    # Stat boosts application (Swords Dance, Quiver Dance, Dragon Dance, Nasty Plot, etc.)
    new_atk_boosts = jnp.clip(
        state.active_boosts[attacker_idx, :5] + jnp.where(target == 1, boosts_5, 0),
        -6,
        6,
    )
    new_def_boosts = jnp.clip(
        state.active_boosts[defender_idx, :5] + jnp.where(target == 0, boosts_5, 0),
        -6,
        6,
    )
    act_boosts = state.active_boosts.at[attacker_idx, :5].set(new_atk_boosts).at[defender_idx, :5].set(new_def_boosts)

    # Status ailment application
    can_status = (state.active_status[defender_idx] == STATUS_NONE) & (status_id > 0) & def_alive
    act_status = state.active_status.at[defender_idx].set(
        jnp.where(can_status, status_id, state.active_status[defender_idx])
    )

    act_curr = state.active_current_hp.at[defender_idx].set(new_curr_hp_def).at[attacker_idx].set(new_curr_hp_atk)
    act_hp = state.active_hp.at[defender_idx].set(new_hp_frac_def).at[attacker_idx].set(new_hp_frac_atk)
    team_hp = state.team_hp.at[defender_idx, 0].set(new_hp_frac_def).at[attacker_idx, 0].set(new_hp_frac_atk)
    team_al = state.team_alive.at[defender_idx, 0].set(def_alive)

    return state.replace(
        active_current_hp=act_curr,
        active_hp=act_hp,
        active_boosts=act_boosts,
        active_status=act_status,
        active_move_pp=pp,
        team_hp=team_hp,
        team_alive=team_al,
    )


def auto_faint_switch(state: BattleState, player_idx: int) -> BattleState:
    needs_switch = state.active_hp[player_idx] <= 0
    alive_bench = state.team_alive[player_idx, 1:6]
    has_alive = jnp.any(alive_bench)

    opp_idx = 1 - player_idx
    opp_t1 = state.active_types[opp_idx, 0]

    bench_sp = state.team_species[player_idx, 1:6]
    bench_t1 = SPECIES_TABLE[bench_sp, 6]
    bench_t2 = SPECIES_TABLE[bench_sp, 7]

    off_eff1 = TYPE_CHART[bench_t1, opp_t1]
    off_eff2 = jnp.where(bench_t2 >= 0, TYPE_CHART[bench_t2, opp_t1], 1.0)
    off_score = off_eff1 + off_eff2

    def_eff1 = TYPE_CHART[opp_t1, bench_t1]
    def_eff2 = jnp.where(bench_t2 >= 0, TYPE_CHART[opp_t1, bench_t2], 1.0)
    def_mult = def_eff1 * def_eff2

    bench_hp = state.team_hp[player_idx, 1:6]
    bench_score = (off_score * 2.0) - (def_mult * 3.0) + (bench_hp * 2.0)
    valid_score = jnp.where(alive_bench, bench_score, -999.0)
    best_bench_slot = jnp.argmax(valid_score) + 1

    def do_switch(s: BattleState) -> BattleState:
        return execute_switch(s, player_idx, best_bench_slot)

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

    pri_0 = jnp.where(
        action_p1 >= 4,
        6,
        MOVE_TABLE[state.active_moves[0, jnp.clip(action_p1, 0, 3)], 4],
    )
    pri_1 = jnp.where(
        action_p2 >= 4,
        6,
        MOVE_TABLE[state.active_moves[1, jnp.clip(action_p2, 0, 3)], 4],
    )

    spe_0 = (
        state.active_stats[0, 5]
        * get_stage_multiplier(state.active_boosts[0, 4])
        * jnp.where(state.active_status[0] == STATUS_PAR, 0.5, 1.0)
    )
    spe_1 = (
        state.active_stats[1, 5]
        * get_stage_multiplier(state.active_boosts[1, 4])
        * jnp.where(state.active_status[1] == STATUS_PAR, 0.5, 1.0)
    )

    p1_first = (
        (pri_0 > pri_1) | ((pri_0 == pri_1) & (spe_0 > spe_1)) | ((pri_0 == pri_1) & (spe_0 == spe_1) & tie_breaker)
    )

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
        dmg_frac = jnp.where(
            st_val == STATUS_BRN,
            1.0 / 16.0,
            jnp.where(st_val == STATUS_PSN, 1.0 / 8.0, 0.0),
        )
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
    boosts_reward = (jnp.sum(s8.active_boosts[0, :5]) - jnp.sum(s8.active_boosts[1, :5])) * 0.02
    status_reward = jnp.where(s8.active_status[1] > 0, 0.05, 0.0) - jnp.where(s8.active_status[0] > 0, 0.05, 0.0)
    win_reward = jnp.where(winner == 1, 1.0, jnp.where(winner == 2, -1.0, 0.0))
    reward = jnp.where(done, win_reward, hp_reward + boosts_reward + status_reward)

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


def sample_battle_teams(rng_key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample diverse 6-mon teams (species and moves) for P1 and P2 from Randbats sets."""
    k1, k2 = jax.random.split(rng_key)
    p1_idx = jax.random.choice(k1, NUM_PROFILES, shape=(6,), replace=False)
    p2_idx = jax.random.choice(k2, NUM_PROFILES, shape=(6,), replace=False)
    return RANDBATS_SPECIES[p1_idx], RANDBATS_SPECIES[p2_idx], RANDBATS_MOVES[p1_idx], RANDBATS_MOVES[p2_idx]


def step_with_autoreset(
    state: BattleState, action_p1: jnp.ndarray, action_p2: jnp.ndarray
) -> Tuple[BattleState, jnp.ndarray, jnp.ndarray]:
    """
    Executes a step in the battle. If the battle reaches a terminal state (done=True),
    returns the terminal reward and automatically resets the state to a fresh battle
    with newly sampled competitive teams.
    """
    next_s, r, d = step(state, action_p1, action_p2)
    k1, k2 = jax.random.split(next_s.rng_key)
    p1_sp, p2_sp, p1_mv, p2_mv = sample_battle_teams(k1)
    reset_s = init_battle(k2, p1_sp, p2_sp, p1_mv, p2_mv)
    final_s = jax.tree_util.tree_map(lambda n, rst: jnp.where(d, rst, n), next_s, reset_s)
    return final_s, r, d


batch_step_autoreset = jax.jit(jax.vmap(step_with_autoreset, in_axes=(0, 0, 0)))


def init_battle_randomized(
    rng_key: jax.Array,
    p1_team: jnp.ndarray,
    p2_team: jnp.ndarray,
    p1_moves: jnp.ndarray,
    p2_moves: jnp.ndarray,
    shuffle_moves: bool = True,
) -> BattleState:
    def _shuffle_moves(moves: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        keys = jax.random.split(key, 6)
        return jax.vmap(lambda m, k: m[jax.random.permutation(k, 4)])(moves, keys)

    k1, k2, k_init = jax.random.split(rng_key, 3)
    p1_m = jax.lax.cond(shuffle_moves, lambda: _shuffle_moves(p1_moves, k1), lambda: p1_moves)
    p2_m = jax.lax.cond(shuffle_moves, lambda: _shuffle_moves(p2_moves, k2), lambda: p2_moves)
    return init_battle(k_init, p1_team, p2_team, p1_m, p2_m)


batch_init_battle_randomized = jax.jit(jax.vmap(init_battle_randomized, in_axes=(0, 0, 0, 0, 0, None)))


def step_with_autoreset_randomized(
    state: BattleState, action_p1: jnp.ndarray, action_p2: jnp.ndarray
) -> Tuple[BattleState, jnp.ndarray, jnp.ndarray]:
    next_s, r, d = step(state, action_p1, action_p2)
    k1, k2 = jax.random.split(next_s.rng_key)
    p1_sp, p2_sp, p1_mv, p2_mv = sample_battle_teams(k1)
    reset_s = init_battle_randomized(k2, p1_sp, p2_sp, p1_mv, p2_mv, True)
    final_s = jax.tree_util.tree_map(lambda n, rst: jnp.where(d, rst, n), next_s, reset_s)
    return final_s, r, d


batch_step_autoreset_randomized = jax.jit(jax.vmap(step_with_autoreset_randomized, in_axes=(0, 0, 0)))
