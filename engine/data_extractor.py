#!/usr/bin/env python3
"""
engine/data_extractor.py
Phase 2: Extracts ground-truth Pokémon Showdown mechanics, species data,
move data, and type chart into compact NumPy arrays for the JAX engine.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from poke_env.data import GenData

# Fixed standard 18 types
TYPE_LIST = [
    "NORMAL",
    "FIRE",
    "WATER",
    "GRASS",
    "ELECTRIC",
    "ICE",
    "FIGHTING",
    "POISON",
    "GROUND",
    "FLYING",
    "PSYCHIC",
    "BUG",
    "ROCK",
    "GHOST",
    "DRAGON",
    "STEEL",
    "DARK",
    "FAIRY",
]
TYPE_TO_IDX: Dict[str, int] = {t: i for i, t in enumerate(TYPE_LIST)}
NUM_TYPES = len(TYPE_LIST)


def build_type_chart(gen_data: GenData) -> np.ndarray:
    """
    Builds an (18, 18) float32 matrix where:
    chart[attacker_idx, defender_idx] = damage multiplier (0.0, 0.5, 1.0, 2.0).
    """
    chart = np.ones((NUM_TYPES, NUM_TYPES), dtype=np.float32)
    raw_chart = gen_data.type_chart

    for def_type, def_idx in TYPE_TO_IDX.items():
        if def_type in raw_chart:
            for atk_type, mult in raw_chart[def_type].items():
                if atk_type in TYPE_TO_IDX:
                    atk_idx = TYPE_TO_IDX[atk_type]
                    chart[atk_idx, def_idx] = float(mult)

    return chart


def extract_species_and_moves(
    gen_data: GenData,
) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
    """
    Extracts species and moves into indexed tables.
    Returns:
      species_to_idx: Dict[str, int]
      move_to_idx: Dict[str, int]
      species_data: np.ndarray shape (N_species, 8) -> [hp, atk, def, spa, spd, spe, type1, type2]
      move_data: np.ndarray shape (N_moves, 6) -> [type, category (0=Phys, 1=Spec, 2=Stat), base_power, accuracy, priority, target]
    """
    pokedex = gen_data.pokedex
    moves = gen_data.moves

    # Species mapping
    species_list = sorted(list(pokedex.keys()))
    species_to_idx = {
        sp: i + 1 for i, sp in enumerate(species_list)
    }  # 0 is reserved for None/Empty
    # Move mapping
    move_list = sorted(list(moves.keys()))
    move_to_idx = {
        mv: i + 1 for i, mv in enumerate(move_list)
    }  # 0 is reserved for None/Empty

    # Build species table: (N_species + 1, 8)
    species_table = np.zeros((len(species_list) + 1, 8), dtype=np.int32)
    for sp, idx in species_to_idx.items():
        data = pokedex[sp]
        bs = data.get("baseStats", {})
        types = data.get("types", ["Normal"])
        t1 = TYPE_TO_IDX.get(types[0].upper(), 0)
        t2 = TYPE_TO_IDX.get(types[1].upper(), -1) if len(types) > 1 else -1

        species_table[idx] = [
            bs.get("hp", 80),
            bs.get("atk", 80),
            bs.get("def", 80),
            bs.get("spa", 80),
            bs.get("spd", 80),
            bs.get("spe", 80),
            t1,
            t2,
        ]

    # Build move table: (N_moves + 1, 6)
    move_table = np.zeros((len(move_list) + 1, 6), dtype=np.int32)
    for mv, idx in move_to_idx.items():
        mdata = moves[mv]
        mtype = TYPE_TO_IDX.get(mdata.get("type", "Normal").upper(), 0)
        cat_str = mdata.get("category", "Physical").upper()
        cat = 0 if cat_str == "PHYSICAL" else (1 if cat_str == "SPECIAL" else 2)
        bp = int(mdata.get("basePower", 0))
        acc = mdata.get("accuracy", 100)
        acc = 100 if acc is True else int(acc) if isinstance(acc, (int, float)) else 100
        pri = int(mdata.get("priority", 0))

        move_table[idx] = [mtype, cat, bp, acc, pri, 0]

    return species_to_idx, move_to_idx, species_table, move_table


def main():
    print("Extracting Gen 9 mechanics and metadata...")
    gen_data = GenData.from_gen(9)

    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    type_chart = build_type_chart(gen_data)
    species_to_idx, move_to_idx, species_table, move_table = extract_species_and_moves(
        gen_data
    )

    out_npz = out_dir / "mechanics_tables.npz"
    np.savez_compressed(
        out_npz,
        type_chart=type_chart,
        species_table=species_table,
        move_table=move_table,
    )

    mapping_path = out_dir / "id_mappings.json"
    with open(mapping_path, "w") as f:
        json.dump(
            {
                "types": TYPE_LIST,
                "species_to_idx": species_to_idx,
                "move_to_idx": move_to_idx,
            },
            f,
        )

    print(f"[✓] Successfully exported mechanics tables to {out_npz}")
    print(f"[✓] Species count: {len(species_to_idx)}, Moves count: {len(move_to_idx)}")
    print(f"[✓] Type chart shape: {type_chart.shape}")


if __name__ == "__main__":
    main()
