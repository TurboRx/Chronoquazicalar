"""
engine/belief_sampler.py
Monte Carlo Belief-State World Sampler for Gen 9 Random Battles.

Implements SOTA imperfect information handling:
In Pokémon Showdown Random Battles, the opponent's unrevealed bench members,
moves, items, abilities, and Tera types are unknown.
This sampler generates N concrete "worlds" (BattleState instances) by sampling
from Bayesian Randbats set distributions (sets.json) and plausibility priors,
allowing simultaneous pUCT search to evaluate moves over the distribution of
possible true game states.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp

from engine.battle_state import (
    WEATHER_NONE,
    WEATHER_RAIN,
    WEATHER_SAND,
    WEATHER_SNOW,
    WEATHER_SUN,
    BattleState,
)
from engine.jax_battle_engine import init_battle
from engine.randbats_knowledge import RandbatsKnowledgeBase, to_id


class BeliefWorldSampler:
    """
    Samples concrete BattleState 'worlds' representing possible complete game states
    consistent with currently observed battlefield information.
    """

    def __init__(
        self,
        kb: Optional[RandbatsKnowledgeBase] = None,
        mappings_path: Optional[Path] = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent
        if mappings_path is None:
            mappings_path = base_dir / "engine" / "data" / "id_mappings.json"

        self.kb = kb or RandbatsKnowledgeBase(mappings_path=mappings_path)

        self.species_to_idx: Dict[str, int] = {}
        self.move_to_idx: Dict[str, int] = {}
        self.type_to_idx: Dict[str, int] = {}

        if mappings_path and mappings_path.exists():
            with open(mappings_path, "r", encoding="utf-8") as f:
                mappings = json.load(f)
                self.species_to_idx = mappings.get("species_to_idx", {})
                self.move_to_idx = mappings.get("move_to_idx", {})
                types_list = mappings.get("types", [])
                self.type_to_idx = {t.upper(): i for i, t in enumerate(types_list)}

        # Filter all_species to only those that exist in our numerical mappings
        self.all_species = [
            sp for sp in self.kb.sets_data.keys() if sp in self.species_to_idx or to_id(sp) in self.species_to_idx
        ]
        if not self.all_species:
            self.all_species = list(self.kb.sets_data.keys())

    def _sample_moves_for_species(
        self,
        species_name: str,
        revealed_moves: Optional[List[str]] = None,
        rng: Optional[random.Random] = None,
    ) -> List[int]:
        r = rng or random
        revealed = [to_id(m) for m in (revealed_moves or []) if m]
        candidate_sets = self.kb.get_candidate_sets(species_name, revealed)

        if candidate_sets:
            # Pick a candidate set at random
            chosen_set = r.choice(candidate_sets)
            pool = [to_id(m) for m in chosen_set.get("movepool", [])]
            # Include revealed moves first
            chosen_moves = list(revealed)
            unrevealed_pool = [m for m in pool if m not in chosen_moves]
            r.shuffle(unrevealed_pool)
            for m in unrevealed_pool:
                if len(chosen_moves) >= 4:
                    break
                chosen_moves.append(m)
        else:
            chosen_moves, _ = self.kb.predict_moves(species_name, revealed)

        while len(chosen_moves) < 4:
            chosen_moves.append("tackle")

        return [self.move_to_idx.get(m, 100) for m in chosen_moves[:4]]

    def sample_world(
        self,
        battle: Any,
        rng: Optional[random.Random] = None,
    ) -> BattleState:
        """
        Constructs a single concrete BattleState from the observable Battle object.
        Unrevealed opponent slots are populated with plausible Randbats species (at 100% HP),
        and unrevealed moves are sampled from the species' Randbats movepool.
        """
        r = rng or random

        # --- Player 1 (Our Team) ---
        act_p1 = battle.active_pokemon
        act_sp_id_1 = self.species_to_idx.get(to_id(act_p1.species) if act_p1 else "pikachu", 950)
        act_hp_1 = float(act_p1.current_hp_fraction if act_p1 else 1.0)

        p1_mvs = [100, 100, 100, 100]
        if act_p1 and act_p1.moves:
            for idx, m in enumerate(list(act_p1.moves.values())[:4]):
                p1_mvs[idx] = self.move_to_idx.get(to_id(m.id), 100)

        p1_bench = [act_sp_id_1]
        p1_bench_hp = [act_hp_1]
        p1_alive = [act_hp_1 > 0]
        p1_moves_list = [p1_mvs]

        for pkm in battle.team.values():
            if pkm != act_p1 and len(p1_bench) < 6:
                sp_id = self.species_to_idx.get(to_id(pkm.species), 1)
                p1_bench.append(sp_id)
                hp_f = float(pkm.current_hp_fraction or 0.0)
                p1_bench_hp.append(hp_f)
                p1_alive.append(hp_f > 0)

                b_mvs = [100, 100, 100, 100]
                if pkm.moves:
                    for idx, m in enumerate(list(pkm.moves.values())[:4]):
                        b_mvs[idx] = self.move_to_idx.get(to_id(m.id), 100)
                p1_moves_list.append(b_mvs)

        while len(p1_bench) < 6:
            p1_bench.append(1)
            p1_bench_hp.append(0.0)
            p1_alive.append(False)
            p1_moves_list.append([100, 100, 100, 100])

        # --- Player 2 (Opponent Team - Imperfect Information Resolution) ---
        act_p2 = battle.opponent_active_pokemon
        act_sp_2_name = to_id(act_p2.species) if act_p2 else "charizard"
        act_sp_id_2 = self.species_to_idx.get(act_sp_2_name, 230)
        act_hp_2 = float(act_p2.current_hp_fraction if act_p2 else 1.0)

        act_p2_rev_moves = [to_id(m.id) for m in act_p2.moves.values()] if act_p2 and act_p2.moves else []
        p2_mvs = self._sample_moves_for_species(act_sp_2_name, act_p2_rev_moves, rng=r)

        p2_bench = [act_sp_id_2]
        p2_bench_hp = [act_hp_2]
        p2_alive = [act_hp_2 > 0]
        p2_moves_list = [p2_mvs]
        used_p2_species = {act_sp_2_name}

        # Revealed bench members
        for pkm in battle.opponent_team.values():
            if pkm != act_p2 and len(p2_bench) < 6:
                sp_name = to_id(pkm.species)
                sp_id = self.species_to_idx.get(sp_name, 1)
                p2_bench.append(sp_id)
                hp_f = float(pkm.current_hp_fraction or 0.0)
                p2_bench_hp.append(hp_f)
                p2_alive.append(hp_f > 0)
                used_p2_species.add(sp_name)

                rev_mvs = [to_id(m.id) for m in pkm.moves.values()] if pkm.moves else []
                bench_mvs = self._sample_moves_for_species(sp_name, rev_mvs, rng=r)
                p2_moves_list.append(bench_mvs)

        # Unrevealed opponent slots: Sample plausible Randbats species (alive at 100% HP)
        remaining_needed = 6 - len(p2_bench)
        available_species = [s for s in self.all_species if s not in used_p2_species]

        if remaining_needed > 0 and available_species:
            sampled_species = r.sample(available_species, min(remaining_needed, len(available_species)))
            for sp_name in sampled_species:
                sp_id = self.species_to_idx.get(sp_name, 1)
                p2_bench.append(sp_id)
                p2_bench_hp.append(1.0)  # Unrevealed Pokémon is 100% healthy
                p2_alive.append(True)

                sp_mvs = self._sample_moves_for_species(sp_name, rng=r)
                p2_moves_list.append(sp_mvs)

        # Pad to 6 if still under (edge case)
        while len(p2_bench) < 6:
            p2_bench.append(1)
            p2_bench_hp.append(0.0)
            p2_alive.append(False)
            p2_moves_list.append([100, 100, 100, 100])

        # Boosts
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

        # Construct JAX BattleState
        rng_key = jax.random.PRNGKey(r.randint(0, 1_000_000))
        base_state = init_battle(
            rng_key,
            jnp.array(p1_bench, dtype=jnp.int32),
            jnp.array(p2_bench, dtype=jnp.int32),
            jnp.array(p1_moves_list, dtype=jnp.int32),
            jnp.array(p2_moves_list, dtype=jnp.int32),
        )

        return base_state.replace(
            active_hp=jnp.array([act_hp_1, act_hp_2], dtype=jnp.float32),
            team_hp=jnp.array([p1_bench_hp, p2_bench_hp], dtype=jnp.float32),
            team_alive=jnp.array([p1_alive, p2_alive], dtype=jnp.bool_),
            active_boosts=jnp.array([boosts_p1, boosts_p2], dtype=jnp.int32),
            weather=jnp.array(weather, dtype=jnp.int32),
            turn_count=jnp.array(battle.turn, dtype=jnp.int32),
        )

    def sample_worlds(
        self,
        battle: Any,
        num_worlds: int = 16,
        seed: Optional[int] = None,
    ) -> List[BattleState]:
        """Samples num_worlds diverse, plausible BattleState instances."""
        r = random.Random(seed) if seed is not None else random.Random()
        return [self.sample_world(battle, rng=r) for _ in range(num_worlds)]
