"""
engine/randbats_knowledge.py
Gen 9 Random Battles Knowledge Base & Set Inference Engine.

Extracts ground-truth set distributions, roles, movepools, abilities,
and Tera types directly from Pokémon Showdown's Gen 9 Randbats dataset.
Performs Bayesian set inference on revealed moves to predict unrevealed
moves, likely Tera types, and exact level scaling for top-tier competitive play.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def to_id(text: str) -> str:
    """Normalize string to Pokémon Showdown ID format (lowercase alphanumeric)."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


class RandbatsKnowledgeBase:
    """
    In-memory knowledge base of all Gen 9 Random Battle sets.
    Enables instant belief-state resolution over opponent moves, abilities, and Tera types.
    """

    def __init__(
        self,
        sets_path: Optional[Path] = None,
        mappings_path: Optional[Path] = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent

        # Locate sets.json: first try pokemon-showdown, fallback to engine/data
        if sets_path is None:
            candidate_showdown = base_dir / "pokemon-showdown" / "data" / "random-battles" / "gen9" / "sets.json"
            candidate_local = base_dir / "engine" / "data" / "randbats_sets.json"
            if candidate_showdown.exists():
                sets_path = candidate_showdown
            elif candidate_local.exists():
                sets_path = candidate_local
            else:
                sets_path = candidate_showdown

        # Locate id_mappings.json
        if mappings_path is None:
            mappings_path = base_dir / "engine" / "data" / "id_mappings.json"

        self.sets_data: Dict[str, dict] = {}
        if sets_path and sets_path.exists():
            with open(sets_path, "r", encoding="utf-8") as f:
                self.sets_data = json.load(f)

            # Also ensure a cached copy in engine/data
            local_cache = base_dir / "engine" / "data" / "randbats_sets.json"
            if not local_cache.exists() and self.sets_data:
                try:
                    local_cache.parent.mkdir(parents=True, exist_ok=True)
                    with open(local_cache, "w", encoding="utf-8") as f:
                        json.dump(self.sets_data, f)
                except Exception:
                    pass

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

    def get_species_entry(self, species_name: str) -> Optional[dict]:
        """Look up raw sets dictionary for a species."""
        sp_id = to_id(species_name)
        return self.sets_data.get(sp_id)

    def get_level(self, species_name: str, default: int = 80) -> int:
        """Get exact Gen 9 Randbats level for a species."""
        entry = self.get_species_entry(species_name)
        if entry and "level" in entry:
            return int(entry["level"])
        return default

    def get_candidate_sets(self, species_name: str, revealed_moves: Optional[List[str]] = None) -> List[dict]:
        """
        Filter possible sets for a species given any revealed moves.
        Returns all sets if no revealed moves match, or matching sets otherwise.
        """
        entry = self.get_species_entry(species_name)
        if not entry or "sets" not in entry:
            return []

        all_sets = entry["sets"]
        if not revealed_moves:
            return all_sets

        norm_revealed = {to_id(m) for m in revealed_moves if m}
        if not norm_revealed:
            return all_sets

        matching_sets = []
        for s in all_sets:
            set_moves = {to_id(m) for m in s.get("movepool", [])}
            if norm_revealed.issubset(set_moves):
                matching_sets.append(s)

        # Fallback to all sets if set constraints are too strict or anomalous
        return matching_sets if matching_sets else all_sets

    def predict_moves(
        self,
        species_name: str,
        revealed_moves: Optional[List[str]] = None,
        max_total: int = 4,
    ) -> Tuple[List[str], List[int]]:
        """
        Predicts complete 4-move set for an opposing Pokémon.
        Combines revealed moves with highest-probability unrevealed moves from sets.json.
        Returns:
            predicted_move_names: List[str] of up to 4 move names/ids
            predicted_move_ids: List[int] of corresponding numerical IDs for JAX engine
        """
        revealed = [to_id(m) for m in (revealed_moves or []) if m]
        candidate_sets = self.get_candidate_sets(species_name, revealed)

        if not candidate_sets:
            # Fallback when species not found in Randbats data
            final_moves = revealed[:max_total]
            while len(final_moves) < max_total:
                final_moves.append("tackle")
            final_ids = [self.move_to_idx.get(m, 100) for m in final_moves]
            return final_moves, final_ids

        # Count frequency of unrevealed moves across candidate sets
        move_counts: Dict[str, int] = {}
        for s in candidate_sets:
            for mv in s.get("movepool", []):
                norm_mv = to_id(mv)
                if norm_mv not in revealed:
                    move_counts[norm_mv] = move_counts.get(norm_mv, 0) + 1

        # Sort unrevealed moves by frequency (most common first)
        sorted_unrevealed = sorted(move_counts.keys(), key=lambda m: move_counts[m], reverse=True)

        # Assemble final 4 moves
        final_moves = list(revealed)
        for mv in sorted_unrevealed:
            if len(final_moves) >= max_total:
                break
            final_moves.append(mv)

        # Ensure exactly max_total moves
        while len(final_moves) < max_total:
            final_moves.append("tackle")

        final_ids = [self.move_to_idx.get(m, 100) for m in final_moves]
        return final_moves, final_ids

    def predict_tera_types(
        self,
        species_name: str,
        revealed_moves: Optional[List[str]] = None,
    ) -> List[str]:
        """Predict likely Tera types for a species given revealed moves."""
        candidate_sets = self.get_candidate_sets(species_name, revealed_moves)
        tera_counts: Dict[str, int] = {}
        for s in candidate_sets:
            for tt in s.get("teraTypes", []):
                tera_counts[tt] = tera_counts.get(tt, 0) + 1

        sorted_tera = sorted(tera_counts.keys(), key=lambda t: tera_counts[t], reverse=True)
        return sorted_tera if sorted_tera else ["Normal"]

    def predict_abilities(
        self,
        species_name: str,
        revealed_moves: Optional[List[str]] = None,
    ) -> List[str]:
        """Predict likely abilities for a species given revealed moves."""
        candidate_sets = self.get_candidate_sets(species_name, revealed_moves)
        ability_counts: Dict[str, int] = {}
        for s in candidate_sets:
            for ab in s.get("abilities", []):
                ability_counts[ab] = ability_counts.get(ab, 0) + 1

        return sorted(ability_counts.keys(), key=lambda a: ability_counts[a], reverse=True)

    def get_candidate_move_scenarios(
        self,
        species_name: str,
        revealed_moves: Optional[List[str]] = None,
        max_scenarios: int = 3,
        max_total: int = 4,
    ) -> List[Tuple[List[str], List[int], float]]:
        """
        Returns distinct move scenarios across candidate set roles for scenario determinization.
        Returns:
            List of (move_names, move_ids, probability_weight)
        """
        revealed = [to_id(m) for m in (revealed_moves or []) if m]
        candidate_sets = self.get_candidate_sets(species_name, revealed)

        if not candidate_sets:
            names, ids = self.predict_moves(species_name, revealed_moves, max_total)
            return [(names, ids, 1.0)]

        selected_sets = candidate_sets[:max_scenarios]
        weight_per_set = 1.0 / len(selected_sets)
        scenarios = []

        for s in selected_sets:
            pool = [to_id(m) for m in s.get("movepool", [])]
            scen_moves = list(revealed)
            for mv in pool:
                if mv not in scen_moves and len(scen_moves) < max_total:
                    scen_moves.append(mv)
            while len(scen_moves) < max_total:
                scen_moves.append("tackle")
            scen_ids = [self.move_to_idx.get(m, 100) for m in scen_moves]
            scenarios.append((scen_moves, scen_ids, weight_per_set))

        return scenarios if scenarios else [(["tackle"] * 4, [100] * 4, 1.0)]

