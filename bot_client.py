"""
Pokémon Showdown player client using poke-env.
Wraps the Transformer policy, heuristic KO checks, and pUCT search.
"""

import argparse
import asyncio
import json
import pickle
from pathlib import Path
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from poke_env.battle.battle import Battle
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
from engine.damage_calc import check_guaranteed_ko
from engine.jax_battle_engine import MOVE_TABLE, TYPE_CHART, init_battle
from engine.randbats_knowledge import RandbatsKnowledgeBase
from models.transformer_policy import ChronosTransformer, state_to_model_inputs
from search.puct_search import PUCTSearchEngine


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

        mappings_path = (
            Path(__file__).resolve().parent / "engine" / "data" / "id_mappings.json"
        )
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
        self.model = ChronosTransformer()
        rng = jax.random.PRNGKey(42)

        dummy_team = jnp.array([950, 230, 150, 1380, 450, 1200], dtype=jnp.int32)
        dummy_moves = jnp.full((6, 4), 100, dtype=jnp.int32)
        dummy_state = init_battle(rng, dummy_team, dummy_team, dummy_moves, dummy_moves)
        sample_inp = state_to_model_inputs(dummy_state, 0)
        batched_sample = {k: v[None, ...] for k, v in sample_inp.items()}
        self.params = self.model.init(rng, batched_sample)

        if checkpoint_path and checkpoint_path.exists():
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
            _ = self.searcher.search(
                dummy_state, max_simulations=1, time_limit_sec=10.0
            )
        except Exception:
            pass

    def _battle_to_battle_state(
        self,
        battle: Battle,
        override_opp_active_moves: Optional[List[int]] = None,
    ) -> BattleState:
        act_p1 = battle.active_pokemon
        act_sp_id_1 = self.species_to_idx.get(
            act_p1.species if act_p1 else "pikachu", 950
        )
        act_hp_1 = act_p1.current_hp_fraction if act_p1 else 1.0

        act_p2 = battle.opponent_active_pokemon
        act_sp_id_2 = self.species_to_idx.get(
            act_p2.species if act_p2 else "charizard", 230
        )
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
            for i, stat in enumerate(
                ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
            ):
                boosts_p1[i] = act_p1.boosts.get(stat, 0)

        boosts_p2 = [0] * 7
        if act_p2 and act_p2.boosts:
            for i, stat in enumerate(
                ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
            ):
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
            p2_revealed = (
                [m.id for m in act_p2.moves.values()] if act_p2 and act_p2.moves else []
            )
            _, p2_mvs = self.kb.predict_moves(
                act_p2.species if act_p2 else "charizard", p2_revealed
            )

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

        return base_state.replace(
            active_hp=jnp.array([act_hp_1, act_hp_2], dtype=jnp.float32),
            active_boosts=jnp.array([boosts_p1, boosts_p2], dtype=jnp.int32),
            weather=jnp.array(weather, dtype=jnp.int32),
            turn_count=jnp.array(battle.turn, dtype=jnp.int32),
        )

    def _should_terastallize(self, battle: Battle, target_move) -> bool:
        """
        Terastallization policy evaluation:
        1. Defensive counter-tera: If opponent threatens super-effective lethal attack and Tera resists or negates it.
        2. Offensive STAB boost: If Tera-boosted STAB turns a 2HKO into a guaranteed 1HKO.
        3. Critical threshold: Low HP preservation on active sweeper.
        """
        if not battle.can_tera or not battle.active_pokemon:
            return False

        act_p1 = battle.active_pokemon
        act_p2 = battle.opponent_active_pokemon

        my_tera = getattr(act_p1, "tera_type", None)
        if my_tera is not None:
            tera_name = my_tera.name if hasattr(my_tera, "name") else str(my_tera)
        else:
            predicted = self.kb.predict_tera_types(act_p1.species)
            tera_name = predicted[0] if predicted else "Normal"
        my_tera_idx = self.type_to_idx.get(tera_name.upper(), 0)

        # 1. Defensive Counter-Tera: Flip unfavorable matchups
        if act_p2:
            p2_revealed = [m.id for m in act_p2.moves.values()] if act_p2.moves else []
            _, opp_mvs = self.kb.predict_moves(act_p2.species, p2_revealed)
            my_t1 = (
                self.type_to_idx.get(act_p1.type_1.name.upper(), 0)
                if act_p1.type_1
                else 0
            )
            my_t2 = (
                self.type_to_idx.get(act_p1.type_2.name.upper(), -1)
                if act_p1.type_2
                else -1
            )

            for m_id in opp_mvs:
                m_type = int(MOVE_TABLE[m_id, 0])
                base_eff = float(TYPE_CHART[m_type, my_t1]) * (
                    float(TYPE_CHART[m_type, my_t2]) if my_t2 >= 0 else 1.0
                )
                tera_eff = float(TYPE_CHART[m_type, my_tera_idx])

                if base_eff >= 2.0 and tera_eff <= 0.5:
                    return True

        # 2. Offensive STAB Tera: Secure lethal game-winning KO
        if target_move:
            mv_type_str = (
                target_move.type.name.upper()
                if hasattr(target_move.type, "name")
                else str(target_move.type).upper()
            )
            mv_type_idx = self.type_to_idx.get(mv_type_str, -1)
            if mv_type_idx == my_tera_idx and act_p2 and act_p2.current_hp_fraction:
                if 0.40 <= act_p2.current_hp_fraction <= 0.85:
                    return True

        # 3. Clutch Survival: Low HP and staying in with lethal attack
        if act_p1.current_hp_fraction and act_p1.current_hp_fraction < 0.45:
            return True

        return False

    def choose_move(self, battle: Battle) -> BattleOrder:
        state = self._battle_to_battle_state(battle)
        act_p1 = battle.active_pokemon
        act_p2 = battle.opponent_active_pokemon
        all_moves = list(act_p1.moves.values()) if act_p1 and act_p1.moves else []
        bench_pkms = [pkm for pkm in battle.team.values() if pkm != act_p1]

        # 1. Guaranteed lethal KO check
        if battle.available_moves:
            has_ko, ko_action = check_guaranteed_ko(state, player_idx=0)
            if bool(has_ko) and int(ko_action) >= 0:
                ko_slot = int(ko_action)
                if ko_slot < len(all_moves):
                    target_move = all_moves[ko_slot]
                    if target_move in battle.available_moves:
                        should_tera = self._should_terastallize(battle, target_move)
                        return self.create_order(target_move, terastallize=should_tera)

        # 2. Search policy resolution with scenario determinization
        opp_sp = act_p2.species if act_p2 else "charizard"
        p2_revealed = (
            [m.id for m in act_p2.moves.values()] if act_p2 and act_p2.moves else []
        )
        scenarios = self.kb.get_candidate_move_scenarios(
            opp_sp, p2_revealed, max_scenarios=2
        )

        if len(scenarios) <= 1:
            chosen_act, strategy, _ = self.searcher.search(
                state,
                time_limit_sec=2.0,
                max_simulations=100,
                temperature=0.3,
            )
        else:
            aggregated_strategy = np.zeros(9, dtype=np.float32)
            time_per_scenario = min(1.0, 2.0 / len(scenarios))
            sims_per_scenario = max(40, 100 // len(scenarios))

            for _, scen_ids, weight in scenarios:
                scen_state = self._battle_to_battle_state(
                    battle, override_opp_active_moves=scen_ids
                )
                _, scen_strat, _ = self.searcher.search(
                    scen_state,
                    time_limit_sec=time_per_scenario,
                    max_simulations=sims_per_scenario,
                    temperature=0.3,
                )
                aggregated_strategy += weight * scen_strat

            strategy = aggregated_strategy / np.sum(aggregated_strategy)
            chosen_act = int(np.argmax(strategy))

        # 3. Forced switch scenario (active Pokémon fainted)
        if not battle.available_moves and battle.available_switches:
            best_switch = None
            best_prob = -1.0
            for idx, pkm in enumerate(bench_pkms):
                if pkm in battle.available_switches:
                    prob = strategy[4 + idx] if (4 + idx) < len(strategy) else 0.0
                    if prob > best_prob:
                        best_prob = prob
                        best_switch = pkm
            if best_switch:
                return self.create_order(best_switch)
            return self.create_order(battle.available_switches[0])

        # 4. Standard move selection (actions 0..3)
        if chosen_act < 4 and battle.available_moves:
            target_move = all_moves[chosen_act] if chosen_act < len(all_moves) else None
            if target_move and target_move in battle.available_moves:
                should_tera = self._should_terastallize(battle, target_move)
                return self.create_order(target_move, terastallize=should_tera)
            # Fallback: best available move by strategy probability
            best_move = battle.available_moves[0]
            best_p = -1.0
            for idx, m in enumerate(all_moves[:4]):
                if m in battle.available_moves and strategy[idx] > best_p:
                    best_p = strategy[idx]
                    best_move = m
            should_tera = self._should_terastallize(battle, best_move)
            return self.create_order(best_move, terastallize=should_tera)

        # 5. Standard switch selection (actions 4..8)
        elif chosen_act >= 4 and battle.available_switches:
            switch_slot = chosen_act - 4
            target_pkm = (
                bench_pkms[switch_slot] if switch_slot < len(bench_pkms) else None
            )
            if target_pkm and target_pkm in battle.available_switches:
                return self.create_order(target_pkm)
            # Fallback: best available switch by strategy probability
            best_switch = battle.available_switches[0]
            best_p = -1.0
            for idx, pkm in enumerate(bench_pkms):
                if pkm in battle.available_switches and (4 + idx) < len(strategy):
                    if strategy[4 + idx] > best_p:
                        best_p = strategy[4 + idx]
                        best_switch = pkm
            return self.create_order(best_switch)

        # 6. Safety fallback
        if battle.available_moves:
            best_move = battle.available_moves[0]
            best_p = -1.0
            for idx, m in enumerate(all_moves[:4]):
                if m in battle.available_moves and strategy[idx] > best_p:
                    best_p = strategy[idx]
                    best_move = m
            should_tera = self._should_terastallize(battle, best_move)
            return self.create_order(best_move, terastallize=should_tera)
        elif battle.available_switches:
            return self.create_order(battle.available_switches[0])
        return self.choose_random_move(battle)


async def run_local_battles(
    num_battles: int = 5, checkpoint_path: Optional[Path] = None
):
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

    print(
        f"Chronos: {chronos.n_won_battles}/{num_battles} wins ({chronos.n_won_battles / num_battles * 100:.1f}%)"
    )
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
        asyncio.run(
            run_local_battles(num_battles=args.battles, checkpoint_path=ckpt_path)
        )


if __name__ == "__main__":
    main()
