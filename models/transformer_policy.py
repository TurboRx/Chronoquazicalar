"""
models/transformer_policy.py
Phase 4: Compact Non-Causal Transformer Policy & Value Network (~8.5M parameters)
Optimized for sub-millisecond inference and dual policy/value predictions.
"""

from typing import Dict, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp

from engine.battle_state import BattleState
from engine.jax_battle_engine import (
    SPECIES_TABLE,
    MOVE_TABLE,
    TYPE_CHART,
    get_valid_actions_mask,
)

NUM_SPECIES = 1605
NUM_MOVES = 1000
NUM_TYPES = 20  # 18 types + None (-1 mapped to 18) + Pad (19)
SEQ_LEN = 18    # CLS(1) + Active(2) + Bench(10) + Moves(4) + Field(1)


class TransformerEncoderBlock(nn.Module):
    """
    Standard Transformer Encoder Layer:
    Multi-Head Self-Attention -> LayerNorm -> Feed-Forward -> LayerNorm.
    """
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Multi-Head Attention
        norm_x = nn.LayerNorm()(x)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            deterministic=deterministic,
        )(norm_x, norm_x)
        x = x + attn_out

        # Feed-Forward Network
        norm_x = nn.LayerNorm()(x)
        ff_out = nn.Dense(self.d_ff)(norm_x)
        ff_out = nn.gelu(ff_out)
        ff_out = nn.Dense(self.d_model)(ff_out)
        x = x + ff_out

        return x


class ChronosTransformer(nn.Module):
    """
    Non-Causal Transformer Encoder (~8.5M parameters) for Gen 9 Random Battles.
    Dual Heads:
      - Policy Head: 9 logits (4 moves + 5 switches)
      - Value Head: Scalar V(s) in [-1.0, 1.0]
    """
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    num_actions: int = 9

    def setup(self):
        # Embeddings
        self.cls_token = self.param("cls_token", nn.initializers.normal(0.02), (1, 1, self.d_model))
        self.species_embed = nn.Embed(NUM_SPECIES, self.d_model)
        self.move_embed = nn.Embed(NUM_MOVES, self.d_model)
        self.type_embed = nn.Embed(NUM_TYPES, self.d_model)
        self.pos_embed = self.param("pos_embed", nn.initializers.normal(0.02), (1, SEQ_LEN, self.d_model))

        # Continuous feature MLPs
        self.active_dense = nn.Sequential([
            nn.Dense(self.d_model),
            nn.gelu,
            nn.Dense(self.d_model),
        ])
        self.bench_dense = nn.Sequential([
            nn.Dense(self.d_model),
            nn.gelu,
            nn.Dense(self.d_model),
        ])
        self.move_dense = nn.Sequential([
            nn.Dense(self.d_model),
            nn.gelu,
            nn.Dense(self.d_model),
        ])
        self.field_dense = nn.Sequential([
            nn.Dense(self.d_model),
            nn.gelu,
            nn.Dense(self.d_model),
        ])

        # 6 Transformer Encoder Blocks
        self.encoder_blocks = [
            TransformerEncoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                name=f"encoder_block_{i}",
            )
            for i in range(self.n_layers)
        ]
        self.final_norm = nn.LayerNorm()

        # Policy Head (256 -> 1024 -> 1024 -> 9)
        self.policy_head = nn.Sequential([
            nn.Dense(self.d_ff),
            nn.gelu,
            nn.Dense(self.d_ff),
            nn.gelu,
            nn.Dense(self.num_actions),
        ])

        # Value Head (256 -> 1024 -> 1024 -> 1)
        self.value_head = nn.Sequential([
            nn.Dense(self.d_ff),
            nn.gelu,
            nn.Dense(self.d_ff),
            nn.gelu,
            nn.Dense(1),
        ])

    def tokenize_battlefield(self, inputs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Converts encoded battlefield dict into a token sequence of shape (B, 18, d_model).
        """
        B = inputs["act_species"].shape[0]

        # Token 0: CLS Token (B, 1, d_model)
        cls_tokens = jnp.repeat(self.cls_token, B, axis=0)

        # Tokens 1..2: Active friendly (1) and opposing (2) Pokémon
        # Species embed + type1/2 embeds + continuous features
        act_sp_emb = self.species_embed(inputs["act_species"])  # (B, 2, d_model)
        act_t1_emb = self.type_embed(inputs["act_types"][:, :, 0])
        act_t2_emb = self.type_embed(inputs["act_types"][:, :, 1])
        act_cont_emb = self.active_dense(inputs["act_continuous"])  # (B, 2, d_model)
        act_tokens = act_sp_emb + act_t1_emb + act_t2_emb + act_cont_emb  # (B, 2, d_model)

        # Tokens 3..7: Friendly Bench (5 tokens)
        bench_sp_0 = self.species_embed(inputs["bench_species_p1"])  # (B, 5, d_model)
        bench_cont_0 = self.bench_dense(inputs["bench_cont_p1"])
        bench_tokens_0 = bench_sp_0 + bench_cont_0

        # Tokens 8..12: Opponent Bench (5 tokens)
        bench_sp_1 = self.species_embed(inputs["bench_species_p2"])  # (B, 5, d_model)
        bench_cont_1 = self.bench_dense(inputs["bench_cont_p2"])
        bench_tokens_1 = bench_sp_1 + bench_cont_1

        # Tokens 13..16: Friendly Active Moves (4 tokens)
        move_emb = self.move_embed(inputs["active_moves"])  # (B, 4, d_model)
        move_type_emb = self.type_embed(inputs["move_types"])
        move_cont_emb = self.move_dense(inputs["move_continuous"])
        move_tokens = move_emb + move_type_emb + move_cont_emb

        # Token 17: Field Token (1 token)
        field_tokens = self.field_dense(inputs["field_features"])[:, None, :]  # (B, 1, d_model)

        # Concatenate into full sequence: (B, 18, d_model)
        tokens = jnp.concatenate(
            [cls_tokens, act_tokens, bench_tokens_0, bench_tokens_1, move_tokens, field_tokens],
            axis=1,
        )
        return tokens + self.pos_embed

    def __call__(
        self,
        inputs: Dict[str, jnp.ndarray],
        valid_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Forward pass.
        Returns:
          policy_logits: (B, 9)
          action_probs: (B, 9)
          value: (B, 1) in [-1.0, 1.0]
        """
        x = self.tokenize_battlefield(inputs)

        # 6 Transformer layers
        for block in self.encoder_blocks:
            x = block(x, deterministic=deterministic)

        x = self.final_norm(x)

        # Use CLS token (index 0) representation
        cls_rep = x[:, 0]  # (B, d_model)

        # Policy & Value outputs
        raw_logits = self.policy_head(cls_rep)  # (B, 9)

        if valid_mask is not None:
            # Mask invalid actions with large negative value
            masked_logits = jnp.where(valid_mask, raw_logits, -1e9)
        else:
            masked_logits = raw_logits

        action_probs = jax.nn.softmax(masked_logits, axis=-1)
        value = jnp.tanh(self.value_head(cls_rep))  # (B, 1)

        return masked_logits, action_probs, value


def state_to_model_inputs(state: BattleState, perspective_player: int = 0) -> Dict[str, jnp.ndarray]:
    """
    Extracts and normalizes features from BattleState into the dict expected by ChronosTransformer.
    Can be batched with jax.vmap.
    """
    p = perspective_player
    opp = 1 - p

    # Active species
    act_species = jnp.array([state.active_species[p], state.active_species[opp]], dtype=jnp.int32)

    # Active types (map -1 to 18)
    p_t2 = jnp.where(state.active_types[p, 1] >= 0, state.active_types[p, 1], 18)
    opp_t2 = jnp.where(state.active_types[opp, 1] >= 0, state.active_types[opp, 1], 18)
    act_types = jnp.array([
        [state.active_types[p, 0], p_t2],
        [state.active_types[opp, 0], opp_t2],
    ], dtype=jnp.int32)

    # Active continuous features: [hp_frac, 7 boosts / 6.0, status_onehot(7), stats/300.0 (6)] -> 21 features
    def make_active_cont(side: int) -> jnp.ndarray:
        hp = state.active_hp[side:side+1]
        boosts = state.active_boosts[side] / 6.0
        st = state.active_status[side]
        st_onehot = jax.nn.one_hot(st, 7)
        stats = state.active_stats[side] / 300.0
        return jnp.concatenate([hp, boosts, st_onehot, stats])

    act_cont = jnp.stack([make_active_cont(p), make_active_cont(opp)])  # (2, 21)

    # Bench features
    bench_sp_p1 = state.team_species[p, 1:6]
    bench_cont_p1 = jnp.stack([state.team_hp[p, 1:6], state.team_alive[p, 1:6].astype(jnp.float32)], axis=-1)  # (5, 2)

    bench_sp_p2 = state.team_species[opp, 1:6]
    bench_cont_p2 = jnp.stack([state.team_hp[opp, 1:6], state.team_alive[opp, 1:6].astype(jnp.float32)], axis=-1)  # (5, 2)

    # Move features
    act_moves = state.active_moves[p]  # (4,)
    m_types = MOVE_TABLE[act_moves, 0]
    m_cats = MOVE_TABLE[act_moves, 1] / 2.0
    m_bps = MOVE_TABLE[act_moves, 2] / 150.0
    m_pris = (MOVE_TABLE[act_moves, 4] + 7.0) / 14.0
    m_pps = state.active_move_pp[p]
    move_cont = jnp.stack([m_cats, m_bps, m_pris, m_pps], axis=-1)  # (4, 4)

    # Field features: weather (5 onehot) + terrain (5 onehot) + hazards (8) -> 18 features
    weather_onehot = jax.nn.one_hot(state.weather, 5)
    terrain_onehot = jax.nn.one_hot(state.terrain, 5)
    hazards_flat = state.hazards.reshape(-1) / 3.0  # (8,)
    field_feat = jnp.concatenate([weather_onehot, terrain_onehot, hazards_flat])

    return {
        "act_species": act_species,
        "act_types": act_types,
        "act_continuous": act_cont,
        "bench_species_p1": bench_sp_p1,
        "bench_cont_p1": bench_cont_p1,
        "bench_species_p2": bench_sp_p2,
        "bench_cont_p2": bench_cont_p2,
        "active_moves": act_moves,
        "move_types": m_types,
        "move_continuous": move_cont,
        "field_features": field_feat,
    }


# Batched converter
batch_state_to_model_inputs = jax.vmap(state_to_model_inputs, in_axes=(0, None))
