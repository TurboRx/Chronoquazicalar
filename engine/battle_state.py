"""
engine/battle_state.py
Immutable JAX PyTree state representation for Gen 9 Random Battles.
"""

import jax.numpy as jnp
from flax import struct

# Status Constants
STATUS_NONE = 0
STATUS_BRN = 1
STATUS_PAR = 2
STATUS_PSN = 3
STATUS_TOX = 4
STATUS_SLP = 5
STATUS_FRZ = 6

# Weather Constants
WEATHER_NONE = 0
WEATHER_SUN = 1
WEATHER_RAIN = 2
WEATHER_SAND = 3
WEATHER_SNOW = 4

# Terrain Constants
TERRAIN_NONE = 0
TERRAIN_ELECTRIC = 1
TERRAIN_GRASSY = 2
TERRAIN_MISTY = 3
TERRAIN_PSYCHIC = 4


@struct.dataclass
class BattleState:
    """
    Fixed-shape immutable PyTree representing the state of a single battle.
    Index 0 corresponds to Player 1, Index 1 corresponds to Player 2.
    """

    # Active Pokémon features
    active_species: jnp.ndarray  # shape (2,), int32
    active_hp: jnp.ndarray  # shape (2,), float32 (fraction [0.0, 1.0])
    active_max_hp: jnp.ndarray  # shape (2,), float32
    active_current_hp: jnp.ndarray  # shape (2,), float32
    active_stats: jnp.ndarray  # shape (2, 6), float32 [HP, Atk, Def, SpA, SpD, Spe]
    active_boosts: (
        jnp.ndarray
    )  # shape (2, 7), int32 [Atk, Def, SpA, SpD, Spe, Acc, Eva] in [-6, 6]
    active_status: jnp.ndarray  # shape (2,), int32
    active_moves: jnp.ndarray  # shape (2, 4), int32
    active_move_pp: jnp.ndarray  # shape (2, 4), float32
    active_types: jnp.ndarray  # shape (2, 2), int32

    # Entire Team features (6 slots per player: slot 0 is active, slots 1-5 are bench)
    team_species: jnp.ndarray  # shape (2, 6), int32
    team_moves: jnp.ndarray  # shape (2, 6, 4), int32
    team_hp: jnp.ndarray  # shape (2, 6), float32
    team_alive: jnp.ndarray  # shape (2, 6), bool

    # Global Battlefield features
    weather: jnp.ndarray  # scalar int32
    weather_turns: jnp.ndarray  # scalar int32
    terrain: jnp.ndarray  # scalar int32
    terrain_turns: jnp.ndarray  # scalar int32
    hazards: jnp.ndarray  # shape (2, 4), int32 [Stealth Rock, Spikes (0-3), Toxic Spikes (0-2), Sticky Web (0-1)]

    # Game Flow
    turn_count: jnp.ndarray  # scalar int32
    rng_key: jnp.ndarray  # shape (2,), uint32
    done: jnp.ndarray  # scalar bool
    winner: jnp.ndarray  # scalar int32 (0 = unfinished, 1 = p1 win, 2 = p2 win)
