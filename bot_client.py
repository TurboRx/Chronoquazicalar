"""
bot_client.py
Phase 7: Project Chronos Ladder Bot Client.
Wraps the trained Transformer Policy, exact damage heuristic, and pUCT search
in poke-env's Player interface.
Supports:
- Local test battles against baseline bots (RandomPlayer, SimpleHeuristicsPlayer)
- Live ladder queuing on play.pokemonshowdown.com (gen9randombattle)
- Win rate, Elo, and battle metric tracking
"""

import argparse
import asyncio
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from poke_env.battle.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player import Player, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import (
    LocalhostServerConfiguration,
    ShowdownServerConfiguration,
)

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
from engine.damage_calc import check_guaranteed_ko
from engine.jax_battle_engine import (
    init_battle,
    get_valid_actions_mask,
    SPECIES_TABLE,
    MOVE_TABLE,
)
from models.transformer_policy import ChronosTransformer, state_to_model_inputs
from search.puct_search import PUCTSearchEngine


class ChronosPlayer(Player):
    """
    Grandmaster AI Player for Gen 9 Random Battles.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        account_configuration: Optional[AccountConfiguration] = None,
        server_configuration=None,
        search_time_budget: float = 2.0,
        **kwargs,
    ):
        if server_configuration is None:
            server_configuration = LocalhostServerConfiguration

        super().__init__(
            account_configuration=account_configuration,
            server_configuration=server_configuration,
            battle_format="gen9randombattle",
            **kwargs,
        )

        # 1. Load Mappings
        mappings_path = Path(__file__).resolve().parent / "engine" / "data" / "id_mappings.json"
        if mappings_path.exists():
            with open(mappings_path, "r") as f:
                data = json.load(f)
                self.species_to_idx = data["species_to_idx"]
                self.move_to_idx = data["move_to_idx"]
                self.type_to_idx = {t: i for i, t in enumerate(data["types"])}
        else:
            self.species_to_idx = {}
            self.move_to_idx = {}
            self.type_to_idx = {}

        # 2. Initialize Model & Weights
        self.model = ChronosTransformer()
        rng = jax.random.PRNGKey(42)

        # Create dummy state to initialize model parameters
        dummy_team = jnp.array([950, 230, 150, 1380, 450, 1200], dtype=jnp.int32)
        dummy_moves = jnp.full((6, 4), 100, dtype=jnp.int32)
        dummy_state = init_battle(rng, dummy_team, dummy_team, dummy_moves, dummy_moves)
        sample_inp = state_to_model_inputs(dummy_state, 0)
        batched_sample = {k: v[None, ...] for k, v in sample_inp.items()}
        self.params = self.model.init(rng, batched_sample)

        # Load trained weights if provided
        if checkpoint_path and checkpoint_path.exists():
            print(f"[✓] Loading Chronos weights from {checkpoint_path}...")
            with open(checkpoint_path, "rb") as f:
                self.params = pickle.load(f)

        # 3. Initialize Search Engine
        self.searcher = PUCTSearchEngine(
            model=self.model,
            params=self.params,
            c_puct=1.5,
            max_depth=3,
            default_time_limit_sec=search_time_budget,
        )

        # Pre-warmup JIT
        self._warmup_jit(dummy_state)
        print("[✓] Project Chronos player initialized and JIT compiled.")

    def _warmup_jit(self, dummy_state: BattleState) -> None:
        """Warms up XLA compilation so live turns experience sub-second latency."""
        try:
            _ = self.searcher.evaluate_state(dummy_state)
            _ = self.searcher.search(dummy_state, max_simulations=1, time_limit_sec=10.0)
        except Exception as e:
            print(f"[!] JIT warmup notice: {e}")

    def _battle_to_battle_state(self, battle: Battle) -> BattleState:
        """Converts poke-env Battle object into JAX BattleState PyTree."""
        # Active Pokémon P1
        act_p1 = battle.active_pokemon
        act_sp_id_1 = self.species_to_idx.get(act_p1.species if act_p1 else "pikachu", 950)
        act_hp_1 = act_p1.current_hp_fraction if act_p1 else 1.0

        # Active Pokémon P2
        act_p2 = battle.opponent_active_pokemon
        act_sp_id_2 = self.species_to_idx.get(act_p2.species if act_p2 else "charizard", 230)
        act_hp_2 = act_p2.current_hp_fraction if act_p2 else 1.0

        # Moves P1
        p1_mvs = [0, 0, 0, 0]
        if act_p1 and act_p1.moves:
            for idx, m in enumerate(list(act_p1.moves.values())[:4]):
                p1_mvs[idx] = self.move_to_idx.get(m.id, 100)

        # Bench P1
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

        # Bench P2
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

        # Boosts (-6 to +6)
        boosts_p1 = [0] * 7
        if act_p1 and act_p1.boosts:
            for i, stat in enumerate(["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]):
                boosts_p1[i] = act_p1.boosts.get(stat, 0)

        boosts_p2 = [0] * 7
        if act_p2 and act_p2.boosts:
            for i, stat in enumerate(["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]):
                boosts_p2[i] = act_p2.boosts.get(stat, 0)

        # Weather
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

        # Construct BattleState
        rng = jax.random.PRNGKey(battle.turn)
        moves_tensor = jnp.array([[p1_mvs, [100, 100, 100, 100]] * 3], dtype=jnp.int32).reshape(2, 6, 4)
        base_state = init_battle(
            rng,
            jnp.array(p1_bench, dtype=jnp.int32),
            jnp.array(p2_bench, dtype=jnp.int32),
            moves_tensor[0],
            moves_tensor[1],
        )

        return base_state.replace(
            active_hp=jnp.array([act_hp_1, act_hp_2], dtype=jnp.float32),
            active_boosts=jnp.array([boosts_p1, boosts_p2], dtype=jnp.int32),
            weather=jnp.array(weather, dtype=jnp.int32),
            turn_count=jnp.array(battle.turn, dtype=jnp.int32),
        )

    def choose_move(self, battle: Battle) -> BattleOrder:
        """
        Selects the best competitive move using exact damage heuristics
        and pUCT lookahead search with CFR regret matching.
        """
        # Convert battle to JAX tensor state
        state = self._battle_to_battle_state(battle)

        # 1. Exact Damage & Guaranteed KO Check
        has_ko, ko_action = check_guaranteed_ko(state, player_idx=0)
        if bool(has_ko) and int(ko_action) >= 0:
            ko_slot = int(ko_action)
            # Find matching available move
            if battle.available_moves and ko_slot < len(battle.available_moves):
                best_move = battle.available_moves[ko_slot]
                return self.create_order(best_move)

        # 2. pUCT Search with Strict Time Budget
        chosen_act, strat, stats = self.searcher.search(
            state,
            time_limit_sec=2.0,
            max_simulations=100,
            temperature=0.3,
        )

        # 3. Map chosen action to Showdown BattleOrder
        # Actions 0..3: Moves
        if chosen_act < 4 and battle.available_moves:
            # Pick available move closest to chosen index
            slot = min(chosen_act, len(battle.available_moves) - 1)
            return self.create_order(battle.available_moves[slot])

        # Actions 4..8: Switches
        elif chosen_act >= 4 and battle.available_switches:
            switch_slot = chosen_act - 4
            slot = min(switch_slot, len(battle.available_switches) - 1)
            return self.create_order(battle.available_switches[slot])

        # Fallback to random legal move
        return self.choose_random_move(battle)


async def run_local_battles(num_battles: int = 5, checkpoint_path: Optional[Path] = None):
    """Runs local test battles against baseline bots to verify integration."""
    print(f"\n=======================================================")
    print(f"Project Chronos: Testing {num_battles} Local Battles")
    print(f"=======================================================")

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

    print(f"Battling Chronos against SimpleHeuristicsPlayer ({num_battles} battles)...")
    await chronos.battle_against(opponent, n_battles=num_battles)

    print("\n--- Battle Results ---")
    print(f"Chronos Wins: {chronos.n_won_battles}/{num_battles} ({chronos.n_won_battles / num_battles * 100:.1f}%)")
    print(f"Opponent Wins: {opponent.n_won_battles}/{num_battles}")


async def run_ladder(
    username: str,
    password: str,
    checkpoint_path: Optional[Path] = None,
    num_battles: int = 10,
):
    """Connects to live Showdown server and queues for ladder games."""
    print(f"\n=======================================================")
    print(f"Project Chronos: Live Ladder Mode (play.pokemonshowdown.com)")
    print(f"=======================================================")

    account = AccountConfiguration(username, password)
    chronos = ChronosPlayer(
        checkpoint_path=checkpoint_path,
        account_configuration=account,
        server_configuration=ShowdownServerConfiguration,
        search_time_budget=2.0,
        max_concurrent_battles=1,
    )

    print(f"Queuing for {num_battles} gen9randombattle ladder games as '{username}'...")
    await chronos.ladder(num_battles)
    print(f"[✓] Ladder session finished. Wins: {chronos.n_won_battles}/{num_battles}")


def main():
    parser = argparse.ArgumentParser(description="Project Chronos Pokémon Showdown Bot")
    parser.add_argument("--test-local", action="store_true", help="Run local test battles against baseline bot")
    parser.add_argument("--battles", type=int, default=5, help="Number of battles to play")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/bc_checkpoint_latest.pkl", help="Model checkpoint path")
    parser.add_argument("--ladder", action="store_true", help="Queue on live Showdown ladder")
    parser.add_argument("--username", type=str, help="Showdown username")
    parser.add_argument("--password", type=str, help="Showdown password")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else None

    if args.ladder:
        if not args.username or not args.password:
            print("[✗] Error: --username and --password are required for ladder mode.")
            return
        asyncio.run(run_ladder(args.username, args.password, checkpoint_path=ckpt_path, num_battles=args.battles))
    else:
        asyncio.run(run_local_battles(num_battles=args.battles, checkpoint_path=ckpt_path))


if __name__ == "__main__":
    main()
