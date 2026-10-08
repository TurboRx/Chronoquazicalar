"""
Pokémon Showdown player client using poke-env.
Wraps the Transformer policy, heuristic KO checks, and pUCT search.
"""

import argparse
import asyncio
import json
import pickle
from pathlib import Path
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from poke_env.battle.battle import Battle
from poke_env.data import GenData
from poke_env.player import Player, SimpleHeuristicsPlayer
from poke_env.player.battle_order import BattleOrder
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import (
    LocalhostServerConfiguration,
    ShowdownServerConfiguration,
)

from engine.battle_state import (
    WEATHER_NONE,
    WEATHER_RAIN,
    WEATHER_SAND,
    WEATHER_SNOW,
    WEATHER_SUN,
    BattleState,
)
from engine.belief_sampler import BeliefWorldSampler
from engine.damage_calc import check_guaranteed_ko
from engine.jax_battle_engine import init_battle
from engine.randbats_knowledge import RandbatsKnowledgeBase, to_id
from models.transformer_policy import ChronosTransformer, state_to_model_inputs
from search.puct_search import PUCTSearchEngine

TABLES_PATH = Path(__file__).resolve().parent / "engine" / "data" / "mechanics_tables.npz"
if TABLES_PATH.exists():
    _tables = np.load(TABLES_PATH)
    MOVE_TABLE = _tables["move_table"]
else:
    MOVE_TABLE = np.zeros((1000, 13), dtype=np.int32)


def get_effective_base_power(move, move_to_idx=None) -> float:
    if not move:
        return 0.0
    bp = getattr(move, "base_power", 0) or 0
    if bp > 0:
        return float(bp)
    if move_to_idx:
        m_id = str(getattr(move, "id", "") or "").lower().replace("-", "").replace(" ", "")
        idx = move_to_idx.get(m_id, 0)
        if idx < len(MOVE_TABLE):
            table_bp = float(MOVE_TABLE[idx][2])
            if table_bp > 0:
                return table_bp
    return 80.0


class ChronosPlayer(Player):
    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        search_time_budget: float = 1.0,
        account_configuration: Optional[AccountConfiguration] = None,
        server_configuration=LocalhostServerConfiguration,
        **kwargs,
    ):
        super().__init__(
            account_configuration=account_configuration,
            server_configuration=server_configuration,
            battle_format="gen9randombattle",
            **kwargs,
        )

        mappings_path = Path(__file__).resolve().parent / "engine" / "data" / "id_mappings.json"
        if mappings_path.exists():
            with open(mappings_path, "r") as f:
                data = json.load(f)
                self.species_to_idx = data["species_to_idx"]
                self.move_to_idx = data["move_to_idx"]
                self.type_to_idx = {t.upper(): i for i, t in enumerate(data["types"])}
        else:
            self.species_to_idx = {}
            self.move_to_idx = {}
            self.type_to_idx = {}

        self.kb = RandbatsKnowledgeBase(mappings_path=mappings_path)
        self.sampler = BeliefWorldSampler(kb=self.kb, mappings_path=mappings_path)
        self.model = ChronosTransformer()
        rng = jax.random.PRNGKey(42)

        dummy_team = jnp.array([950, 230, 150, 1380, 450, 1200], dtype=jnp.int32)
        dummy_moves = jnp.full((6, 4), 100, dtype=jnp.int32)
        dummy_state = init_battle(rng, dummy_team, dummy_team, dummy_moves, dummy_moves)
        sample_inp = state_to_model_inputs(dummy_state, 0)
        batched_sample = {k: v[None, ...] for k, v in sample_inp.items()}
        self.params = self.model.init(rng, batched_sample)

        if checkpoint_path and Path(checkpoint_path).exists():
            with open(checkpoint_path, "rb") as f:
                self.params = pickle.load(f)

        self.searcher = PUCTSearchEngine(
            model=self.model,
            params=self.params,
            c_puct=1.5,
            max_depth=3,
            default_time_limit_sec=search_time_budget,
        )

        self._warmup_jit(dummy_state)

    def _warmup_jit(self, dummy_state: BattleState) -> None:
        try:
            _ = self.searcher.evaluate_state(dummy_state)
            _ = self.searcher.search(dummy_state, max_simulations=1, time_limit_sec=10.0)
        except Exception:
            pass

    def _battle_to_battle_state(
        self,
        battle: Battle,
        override_opp_active_moves: Optional[List[int]] = None,
    ) -> BattleState:
        act_p1 = battle.active_pokemon
        act_sp_id_1 = self.species_to_idx.get(act_p1.species if act_p1 else "pikachu", 950)
        act_hp_1 = act_p1.current_hp_fraction if act_p1 else 1.0

        act_p2 = battle.opponent_active_pokemon
        act_sp_id_2 = self.species_to_idx.get(act_p2.species if act_p2 else "charizard", 230)
        act_hp_2 = act_p2.current_hp_fraction if act_p2 else 1.0

        p1_mvs = [0, 0, 0, 0]
        if act_p1 and act_p1.moves:
            for idx, m in enumerate(list(act_p1.moves.values())[:4]):
                p1_mvs[idx] = self.move_to_idx.get(m.id, 100)

        p1_bench = [act_sp_id_1]
        p1_bench_hp = [act_hp_1]
        p1_alive = [act_hp_1 > 0]
        for pkm in battle.team.values():
            if pkm != act_p1 and len(p1_bench) < 6:
                sp_id = self.species_to_idx.get(pkm.species, 1)
                p1_bench.append(sp_id)
                hp_f = pkm.current_hp_fraction or 0.0
                p1_bench_hp.append(hp_f)
                p1_alive.append(hp_f > 0)
        while len(p1_bench) < 6:
            p1_bench.append(1)
            p1_bench_hp.append(0.0)
            p1_alive.append(False)

        p2_bench = [act_sp_id_2]
        p2_bench_hp = [act_hp_2]
        p2_alive = [act_hp_2 > 0]
        for pkm in battle.opponent_team.values():
            if pkm != act_p2 and len(p2_bench) < 6:
                sp_id = self.species_to_idx.get(pkm.species, 1)
                p2_bench.append(sp_id)
                hp_f = pkm.current_hp_fraction or 0.0
                p2_bench_hp.append(hp_f)
                p2_alive.append(hp_f > 0)
        while len(p2_bench) < 6:
            p2_bench.append(1)
            p2_bench_hp.append(0.0)
            p2_alive.append(False)

        boosts_p1 = [0] * 7
        if act_p1 and act_p1.boosts:
            for i, stat in enumerate(["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]):
                boosts_p1[i] = act_p1.boosts.get(stat, 0)

        boosts_p2 = [0] * 7
        if act_p2 and act_p2.boosts:
            for i, stat in enumerate(["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]):
                boosts_p2[i] = act_p2.boosts.get(stat, 0)

        weather = WEATHER_NONE
        if battle.weather:
            w_str = list(battle.weather.keys())[0].name
            if "SUN" in w_str:
                weather = WEATHER_SUN
            elif "RAIN" in w_str:
                weather = WEATHER_RAIN
            elif "SAND" in w_str:
                weather = WEATHER_SAND
            elif "SNOW" in w_str:
                weather = WEATHER_SNOW

        rng = jax.random.PRNGKey(battle.turn)
        p1_moves_all = jnp.array([p1_mvs] + [[100, 100, 100, 100]] * 5, dtype=jnp.int32)

        # Opponent active moves: revealed + predicted via RandbatsKnowledgeBase (or scenario override)
        if override_opp_active_moves is not None:
            p2_mvs = override_opp_active_moves
        else:
            p2_revealed = [m.id for m in act_p2.moves.values()] if act_p2 and act_p2.moves else []
            _, p2_mvs = self.kb.predict_moves(act_p2.species if act_p2 else "charizard", p2_revealed)

        # Opponent team moves
        p2_team_moves = [p2_mvs]
        for pkm in battle.opponent_team.values():
            if pkm != act_p2 and len(p2_team_moves) < 6:
                bench_revealed = [m.id for m in pkm.moves.values()] if pkm.moves else []
                _, b_mvs = self.kb.predict_moves(pkm.species, bench_revealed)
                p2_team_moves.append(b_mvs)
        while len(p2_team_moves) < 6:
            p2_team_moves.append([100, 100, 100, 100])

        p2_moves_all = jnp.array(p2_team_moves, dtype=jnp.int32)
        base_state = init_battle(
            rng,
            jnp.array(p1_bench, dtype=jnp.int32),
            jnp.array(p2_bench, dtype=jnp.int32),
            p1_moves_all,
            p2_moves_all,
        )

        p1_stats = base_state.active_stats[0]
        if act_p1 and getattr(act_p1, "stats", None):
            p1_stats = jnp.array(
                [
                    act_p1.max_hp if act_p1.max_hp else p1_stats[0],
                    act_p1.stats.get("atk") if act_p1.stats.get("atk") is not None else p1_stats[1],
                    act_p1.stats.get("def") if act_p1.stats.get("def") is not None else p1_stats[2],
                    act_p1.stats.get("spa") if act_p1.stats.get("spa") is not None else p1_stats[3],
                    act_p1.stats.get("spd") if act_p1.stats.get("spd") is not None else p1_stats[4],
                    act_p1.stats.get("spe") if act_p1.stats.get("spe") is not None else p1_stats[5],
                ],
                dtype=jnp.float32,
            )

        p2_stats = base_state.active_stats[1]
        if act_p2 and getattr(act_p2, "base_stats", None):
            p2_hp = act_p2.max_hp if hasattr(act_p2, "max_hp") and act_p2.max_hp else p2_stats[0]
            p2_stats = jnp.array(
                [
                    p2_hp,
                    act_p2.base_stats.get("atk") if act_p2.base_stats.get("atk") is not None else p2_stats[1],
                    act_p2.base_stats.get("def") if act_p2.base_stats.get("def") is not None else p2_stats[2],
                    act_p2.base_stats.get("spa") if act_p2.base_stats.get("spa") is not None else p2_stats[3],
                    act_p2.base_stats.get("spd") if act_p2.base_stats.get("spd") is not None else p2_stats[4],
                    act_p2.base_stats.get("spe") if act_p2.base_stats.get("spe") is not None else p2_stats[5],
                ],
                dtype=jnp.float32,
            )

        active_stats = jnp.array([p1_stats, p2_stats])

        p1_curr_hp = (
            act_p1.current_hp
            if act_p1 and hasattr(act_p1, "current_hp") and act_p1.current_hp is not None
            else base_state.active_current_hp[0]
        )
        p1_max_hp = (
            act_p1.max_hp if act_p1 and hasattr(act_p1, "max_hp") and act_p1.max_hp else base_state.active_max_hp[0]
        )
        p2_max_hp = p2_stats[0]
        p2_curr_hp = act_hp_2 * p2_max_hp

        return base_state.replace(
            active_hp=jnp.array([act_hp_1, act_hp_2], dtype=jnp.float32),
            active_current_hp=jnp.array([p1_curr_hp, p2_curr_hp], dtype=jnp.float32),
            active_max_hp=jnp.array([p1_max_hp, p2_max_hp], dtype=jnp.float32),
            active_stats=active_stats,
            active_boosts=jnp.array([boosts_p1, boosts_p2], dtype=jnp.int32),
            weather=jnp.array(weather, dtype=jnp.int32),
            turn_count=jnp.array(battle.turn, dtype=jnp.int32),
        )

    def _get_effective_multiplier(self, defender, move) -> float:
        if not defender or not move:
            return 1.0
        try:
            eff = float(defender.damage_multiplier(move))
        except Exception:
            eff = 1.0

        # Ability immunity resolution: check revealed or predicted Randbats ability
        m_type = move.type.name.upper() if (hasattr(move, "type") and move.type) else ""
        ab_name = getattr(defender, "ability", None)
        if ab_name:
            candidate_abs = [str(ab_name)]
        elif hasattr(defender, "species") and defender.species:
            candidate_abs = [str(a) for a in self.kb.predict_abilities(defender.species)]
        else:
            candidate_abs = []

        norm_abs = {to_id(a) for a in candidate_abs}
        if m_type == "GROUND" and norm_abs.intersection({"levitate", "eartheater"}):
            return 0.0
        if m_type == "FIRE" and norm_abs.intersection({"flashfire", "wellbakedbody"}):
            return 0.0
        if m_type == "WATER" and norm_abs.intersection({"waterabsorb", "stormdrain", "dryskin"}):
            return 0.0
        if m_type == "ELECTRIC" and norm_abs.intersection({"voltabsorb", "lightningrod", "motordrive"}):
            return 0.0
        if m_type == "GRASS" and "sapsipper" in norm_abs:
            return 0.0

        return eff

    def _should_terastallize(self, battle: Battle, target_move, state: BattleState) -> bool:
        """
        Terastallization evaluation:
        1. Defensive Tera: Turn incoming lethal/super-effective weakness into resistance.
        2. Offensive Tera: Move matches Tera type and deals super-effective/neutral STAB damage.
        """
        if not battle.can_tera or not battle.active_pokemon:
            return False

        act_p1 = battle.active_pokemon
        my_tera = getattr(act_p1, "tera_type", None)
        if my_tera is not None:
            tera_name = my_tera.name if hasattr(my_tera, "name") else str(my_tera)
        else:
            predicted = self.kb.predict_tera_types(act_p1.species)
            tera_name = predicted[0] if predicted else "Normal"

        tera_type_upper = str(tera_name).upper()
        opp = battle.opponent_active_pokemon

        # 1. Defensive Tera: Flip incoming weakness into resistance when in danger
        if opp:
            my_hp = act_p1.current_hp_fraction if act_p1.current_hp_fraction is not None else 1.0
            if my_hp <= 0.65:
                type_chart = GenData.from_gen(9).type_chart
                opp_moves = list(opp.moves.values()) if opp.moves else []
                opp_types = [t for t in (opp.types or []) if t]
                for om in opp_moves or opp_types:
                    m_type = (
                        om.type.name.upper()
                        if hasattr(om, "type") and om.type
                        else (om.name.upper() if hasattr(om, "name") else str(om).upper())
                    )
                    if m_type:
                        act_eff = self._get_effective_multiplier(act_p1, om) if hasattr(om, "type") else 1.0
                        tera_mult = float(type_chart.get(tera_type_upper, {}).get(m_type, 1.0))
                        if act_eff >= 2.0 and tera_mult <= 1.0:
                            return True

        # 2. Offensive Tera: Move matches Tera type and deals super-effective/neutral damage
        if target_move and hasattr(target_move, "type") and target_move.type:
            move_type_name = target_move.type.name.upper()
            if move_type_name == tera_type_upper:
                if opp:
                    multiplier = self._get_effective_multiplier(opp, target_move)
                    if multiplier >= 1.0:
                        return True

        return False

    def _evaluate_move_tactics(self, battle: Battle, move) -> Tuple[float, str]:
        if not move or not battle.opponent_active_pokemon or not battle.active_pokemon:
            return 0.0, "invalid"

        opp = battle.opponent_active_pokemon
        act = battle.active_pokemon
        my_hp = act.current_hp_fraction if act and act.current_hp_fraction is not None else 1.0
        opp_hp = opp.current_hp_fraction if opp and opp.current_hp_fraction is not None else 1.0

        m_id = str(getattr(move, "id", "") or "").lower().replace("-", "").replace(" ", "")
        m_idx = self.move_to_idx.get(m_id, 0)
        mdata = MOVE_TABLE[m_idx] if m_idx < len(MOVE_TABLE) else np.zeros(13, dtype=np.int32)
        # mdata: [mtype, cat, bp, acc, pri, target, heal_pct, status_id, b_atk, b_def, b_spa, b_spd, b_spe]
        m_cat = int(mdata[1])
        m_bp = float(mdata[2]) if int(mdata[2]) > 0 else (getattr(move, "base_power", 0) or 0.0)
        target = int(mdata[5])
        heal_pct = int(mdata[6])
        status_id = int(mdata[7])
        b_atk, b_def, b_spa, b_spd, b_spe = [int(x) for x in mdata[8:13]]
        boost_sum = b_atk + b_def + b_spa + b_spd + b_spe

        opp_types = [t.name.upper() for t in (opp.types or []) if t]
        my_types = [t for t in (act.types or []) if t]
        weather_str = str(battle.weather).lower() if battle.weather else ""

        # 1. Healing moves (heal_pct > 0)
        if heal_pct > 0:
            if my_hp >= 0.75:
                return 0.0, "heal_full_hp"
            if my_hp <= 0.40:
                return 285.0, "urgent_recovery"
            if my_hp <= 0.60:
                return 235.0, "recovery"
            return 90.0, "soft_recovery"

        # 2. Setup moves (target == 1 and positive stat boosts)
        if target == 1 and boost_sum > 0:
            if my_hp < 0.50:
                return 0.0, "setup_too_low_hp"
            my_boosts = act.boosts if (act and act.boosts) else {}
            curr_atk_b = my_boosts.get("atk", 0)
            curr_spa_b = my_boosts.get("spa", 0)
            if max(curr_atk_b, curr_spa_b) >= 2:
                return 30.0, "setup_already_boosted"
            if my_hp >= 0.65:
                return 280.0, "setup_boost"
            return 170.0, "moderate_setup"

        # 3. Status ailment moves (status_id > 0)
        if status_id > 0:
            if opp.status is not None:
                return 0.0, "already_statused"
            # 5 = SLP (Sleep)
            if status_id == 5:
                if "GRASS" in opp_types:
                    return 0.0, "grass_powder_immune"
                return 320.0, "inflict_sleep"
            # 2 = PAR (Paralysis)
            if status_id == 2:
                if "ELECTRIC" in opp_types or "GROUND" in opp_types:
                    return 0.0, "paralysis_immune"
                return 240.0, "inflict_paralysis"
            # 1 = BRN (Burn)
            if status_id == 1:
                if "FIRE" in opp_types:
                    return 0.0, "burn_immune"
                return 220.0, "inflict_burn"
            # 3, 4 = PSN / TOX
            if status_id in (3, 4):
                if "POISON" in opp_types or "STEEL" in opp_types:
                    return 0.0, "toxic_immune"
                return 160.0, "inflict_toxic"

        # 4. Damaging moves
        if m_bp > 0:
            eff = self._get_effective_multiplier(opp, move)
            if eff == 0.0:
                return -100.0, "immune_attack"

            stab = 1.5 if (move.type and move.type in my_types) else 1.0

            weather_mod = 1.0
            if "sun" in weather_str:
                if move.type and move.type.name.upper() == "FIRE":
                    weather_mod = 1.5
                elif move.type and move.type.name.upper() == "WATER":
                    weather_mod = 0.5
            elif "rain" in weather_str:
                if move.type and move.type.name.upper() == "WATER":
                    weather_mod = 1.5
                elif move.type and move.type.name.upper() == "FIRE":
                    weather_mod = 0.5

            stat_mult = 1.0
            if m_cat == 0:  # Physical
                atk_b = act.boosts.get("atk", 0) if (act and act.boosts) else 0
                atk_m = (2.0 + atk_b) / 2.0 if atk_b >= 0 else 2.0 / (2.0 - atk_b)
                def_b = opp.boosts.get("def", 0) if (opp and opp.boosts) else 0
                def_m = (2.0 + def_b) / 2.0 if def_b >= 0 else 2.0 / (2.0 - def_b)
                stat_mult = atk_m / max(def_m, 0.1)
            elif m_cat == 1:  # Special
                spa_b = act.boosts.get("spa", 0) if (act and act.boosts) else 0
                spa_m = (2.0 + spa_b) / 2.0 if spa_b >= 0 else 2.0 / (2.0 - spa_b)
                spd_b = opp.boosts.get("spd", 0) if (opp and opp.boosts) else 0
                spd_m = (2.0 + spd_b) / 2.0 if spd_b >= 0 else 2.0 / (2.0 - spd_b)
                stat_mult = spa_m / max(spd_m, 0.1)

            dmg_estimate = m_bp * eff * stab * weather_mod * stat_mult

            # Lethal KO check
            is_lethal = False
            if eff >= 2.0 and m_bp >= 80.0 and opp_hp <= 0.65:
                is_lethal = True
            elif dmg_estimate >= 120.0 and opp_hp <= 0.40:
                is_lethal = True
            elif opp_hp <= 0.25 and eff >= 1.0 and m_bp >= 40.0:
                is_lethal = True
            elif opp_hp <= 0.15 and eff >= 0.5 and m_bp >= 40.0:
                is_lethal = True

            if is_lethal:
                return 350.0 + dmg_estimate, "lethal_ko"

            return dmg_estimate, "damaging_attack"

        return 20.0, "other_move"

    def _get_best_heuristic_move(self, battle: Battle):
        if not battle.available_moves:
            return None, 0.0, False, "no_moves"

        best_move = battle.available_moves[0]
        best_score = -1e9
        best_reason = "default"
        is_lethal = False

        for move in battle.available_moves:
            score, reason = self._evaluate_move_tactics(battle, move)
            if score > best_score:
                best_score = score
                best_move = move
                best_reason = reason
                is_lethal = reason == "lethal_ko"

        return best_move, best_score, is_lethal, best_reason

    def _get_best_switch(self, battle: Battle):
        best_pkm, _ = self._get_best_switch_with_score(battle)
        return best_pkm

    def _get_best_switch_with_score(self, battle: Battle) -> Tuple[Optional[Any], float]:
        if not battle.available_switches:
            return None, -1e9
        if not battle.opponent_active_pokemon:
            return battle.available_switches[0], 50.0

        opp = battle.opponent_active_pokemon
        opp_moves = list(opp.moves.values()) if opp.moves else []
        opp_types = [t for t in (opp.types or []) if t]

        best_pkm = battle.available_switches[0]
        best_score = -1e9

        for pkm in battle.available_switches:
            if not pkm or pkm.fainted:
                continue

            pkm_moves = list(pkm.moves.values()) if pkm.moves else []
            pkm_types = [t for t in (pkm.types or []) if t]

            # 1. Offensive score: best move pkm has against opponent
            off_score = 0.0
            for m in pkm_moves:
                bp = get_effective_base_power(m)
                if bp > 0:
                    eff = self._get_effective_multiplier(opp, m)
                    stab = 1.5 if (m.type and m.type in pkm_types) else 1.0
                    dmg = bp * eff * stab
                    if dmg > off_score:
                        off_score = dmg
            if not pkm_moves or off_score == 0.0:
                off_score = 60.0

            # 2. Defensive multiplier: maximum multiplier opp can hit pkm with
            def_mult = 1.0
            has_immunity = False
            if opp_moves:
                for om in opp_moves:
                    bp = get_effective_base_power(om)
                    if bp > 0:
                        m_eff = self._get_effective_multiplier(pkm, om)
                        if m_eff == 0.0:
                            has_immunity = True
                        elif m_eff > def_mult:
                            def_mult = m_eff
            else:
                for ot in opp_types:
                    try:
                        t_eff = float(pkm.damage_multiplier(ot))
                        if t_eff == 0.0:
                            has_immunity = True
                        elif t_eff > def_mult:
                            def_mult = t_eff
                    except Exception:
                        pass

            # 3. Speed advantage
            pkm_spe = (
                (pkm.stats.get("spe") if getattr(pkm, "stats", None) else None)
                or (pkm.base_stats.get("spe") if getattr(pkm, "base_stats", None) else 100)
                or 100
            )
            opp_spe = (
                (opp.stats.get("spe") if getattr(opp, "stats", None) else None)
                or (opp.base_stats.get("spe") if getattr(opp, "base_stats", None) else 100)
                or 100
            )
            speed_bonus = 30.0 if float(pkm_spe) > float(opp_spe) else 0.0

            # 4. HP bonus
            hp_frac = pkm.current_hp_fraction if pkm.current_hp_fraction is not None else 1.0
            hp_bonus = hp_frac * 50.0
            immune_bonus = 100.0 if has_immunity else 0.0

            score = off_score - (def_mult * 70.0) + speed_bonus + hp_bonus + immune_bonus
            if score > best_score:
                best_score = score
                best_pkm = pkm

        return best_pkm, best_score

    def choose_move(self, battle: Battle) -> BattleOrder:
        state = self._battle_to_battle_state(battle)
        act_p1 = battle.active_pokemon
        all_moves = list(act_p1.moves.values()) if act_p1 and act_p1.moves else []

        # 1. Guaranteed lethal KO check via JAX engine damage calc
        if battle.available_moves:
            has_ko, ko_action = check_guaranteed_ko(state, player_idx=0)
            if bool(has_ko) and int(ko_action) >= 0:
                ko_slot = int(ko_action)
                if ko_slot < len(all_moves):
                    target_move = all_moves[ko_slot]
                    if target_move in battle.available_moves:
                        should_tera = self._should_terastallize(battle, target_move, state)
                        return self.create_order(target_move, terastallize=should_tera)

        # 2. Tactical heuristic analysis (immediate priority moves)
        heuristic_move = None
        heuristic_score = 0.0
        is_lethal = False
        h_reason = ""
        if battle.available_moves:
            heuristic_move, heuristic_score, is_lethal, h_reason = self._get_best_heuristic_move(battle)
            # If guaranteed lethal KO or urgent high-priority move (e.g. anti-setup haze or emergency healing)
            if is_lethal or heuristic_score >= 210.0:
                if heuristic_move in battle.available_moves:
                    should_tera = self._should_terastallize(battle, heuristic_move, state)
                    return self.create_order(heuristic_move, terastallize=should_tera)

        # 3. Monte Carlo Belief-State Sampling across imperfect-information worlds
        revealed_opp_count = len([pkm for pkm in battle.opponent_team.values() if pkm.species])

        # Dynamic world count & budget: Fast on early game, deep on mid/endgames
        if revealed_opp_count <= 2:
            num_worlds = 4
            time_budget = 0.5
            sims_per_world = 30
        else:
            num_worlds = 6
            time_budget = 1.2
            sims_per_world = 40

        sampled_worlds = self.sampler.sample_worlds(battle, num_worlds=num_worlds)
        chosen_act, strategy, search_stats = self.searcher.search_belief_worlds(
            world_states=sampled_worlds,
            time_limit_sec=time_budget,
            sims_per_world=sims_per_world,
            temperature=0.3,
            revealed_opp_count=revealed_opp_count,
        )

        searched_sims = (
            search_stats.get("searched_simulations", 0)
            if isinstance(search_stats, dict)
            else getattr(search_stats, "searched_simulations", 0)
        )

        # If search was too shallow and we have a strong heuristic move
        if searched_sims < 12 and heuristic_move and heuristic_score > 60.0:
            should_tera = self._should_terastallize(battle, heuristic_move, state)
            return self.create_order(heuristic_move, terastallize=should_tera)

        # 4. Standard move selection (actions 0..3)
        if chosen_act < 4 and battle.available_moves:
            mcts_move = all_moves[chosen_act] if chosen_act < len(all_moves) else None
            if mcts_move and mcts_move in battle.available_moves:
                m_score, _ = self._evaluate_move_tactics(battle, mcts_move)
                # Override if MCTS picked an immune attack or severely inferior option
                if m_score < 0.0 or (heuristic_move and heuristic_score >= 1.6 * m_score and heuristic_score > 80.0):
                    target_move = heuristic_move
                else:
                    target_move = mcts_move
                should_tera = self._should_terastallize(battle, target_move, state)
                return self.create_order(target_move, terastallize=should_tera)

            # Fallback: heuristic move or best available move
            if heuristic_move and heuristic_move in battle.available_moves:
                should_tera = self._should_terastallize(battle, heuristic_move, state)
                return self.create_order(heuristic_move, terastallize=should_tera)
            return self.create_order(battle.available_moves[0])

        # 5. Switch selection (actions 4..8 or forced switch)
        if battle.available_switches:
            best_switch = self._get_best_switch(battle)
            if best_switch and best_switch in battle.available_switches:
                return self.create_order(best_switch)
            return self.create_order(battle.available_switches[0])

        # 6. Safety fallback
        if battle.available_moves:
            best_move = battle.available_moves[0]
            should_tera = self._should_terastallize(battle, best_move, state)
            return self.create_order(best_move, terastallize=should_tera)
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])
        return self.choose_random_move(battle)


async def run_local_battles(num_battles: int = 5, checkpoint_path: Optional[Path] = None):
    chronos = ChronosPlayer(
        checkpoint_path=checkpoint_path,
        search_time_budget=1.0,
        max_concurrent_battles=1,
    )
    opponent = SimpleHeuristicsPlayer(
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen9randombattle",
        max_concurrent_battles=1,
    )

    await chronos.battle_against(opponent, n_battles=num_battles)

    print(f"Chronos: {chronos.n_won_battles}/{num_battles} wins ({chronos.n_won_battles / num_battles * 100:.1f}%)")
    print(f"Opponent: {opponent.n_won_battles}/{num_battles} wins")


async def run_ladder(
    username: str,
    password: str,
    checkpoint_path: Optional[Path] = None,
    num_battles: int = 10,
):
    account = AccountConfiguration(username, password)
    chronos = ChronosPlayer(
        checkpoint_path=checkpoint_path,
        account_configuration=account,
        server_configuration=ShowdownServerConfiguration,
        search_time_budget=2.0,
        max_concurrent_battles=1,
    )

    await chronos.ladder(num_battles)
    print(f"Completed {num_battles} games. Wins: {chronos.n_won_battles}/{num_battles}")


def main():
    parser = argparse.ArgumentParser(description="Pokémon Showdown bot client")
    parser.add_argument(
        "--test-local",
        action="store_true",
        help="Run local battles against baseline player",
    )
    parser.add_argument("--battles", type=int, default=5, help="Number of battles")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/bc_checkpoint_latest.pkl",
        help="Path to checkpoint",
    )
    parser.add_argument("--ladder", action="store_true", help="Queue on public ladder")
    parser.add_argument("--username", type=str, help="Showdown account username")
    parser.add_argument("--password", type=str, help="Showdown account password")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else None

    if args.ladder:
        if not args.username or not args.password:
            print("Error: --username and --password are required for ladder mode")
            return
        asyncio.run(
            run_ladder(
                args.username,
                args.password,
                checkpoint_path=ckpt_path,
                num_battles=args.battles,
            )
        )
    else:
        asyncio.run(run_local_battles(num_battles=args.battles, checkpoint_path=ckpt_path))


if __name__ == "__main__":
    main()
