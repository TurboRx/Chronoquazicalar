"""
Non-causal Transformer policy and value network in Flax.
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
NUM_TYPES = 20
SEQ_LEN = 18


class TransformerEncoderBlock(nn.Module):
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        norm_x = nn.LayerNorm()(x)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            deterministic=deterministic,
        )(norm_x, norm_x)
        x = x + attn_out

        norm_x = nn.LayerNorm()(x)
        ff_out = nn.Dense(self.d_ff)(norm_x)
        ff_out = nn.gelu(ff_out)
        ff_out = nn.Dense(self.d_model)(ff_out)
        return x + ff_out


class ChronosTransformer(nn.Module):
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    num_actions: int = 9

    def setup(self):
        self.cls_token = self.param("cls_token", nn.initializers.normal(0.02), (1, 1, self.d_model))
        self.species_embed = nn.Embed(NUM_SPECIES, self.d_model)
        self.move_embed = nn.Embed(NUM_MOVES, self.d_model)
        self.type_embed = nn.Embed(NUM_TYPES, self.d_model)
        self.pos_embed = self.param("pos_embed", nn.initializers.normal(0.02), (1, SEQ_LEN, self.d_model))

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

        self.policy_head = nn.Sequential([
            nn.Dense(self.d_ff),
            nn.gelu,
            nn.Dense(self.d_ff),
            nn.gelu,
            nn.Dense(self.num_actions),
        ])

        self.value_head = nn.Sequential([
            nn.Dense(self.d_ff),
            nn.gelu,
            nn.Dense(self.d_ff),
            nn.gelu,
            nn.Dense(1),
        ])

    def tokenize_battlefield(self, inputs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        B = inputs["act_species"].shape[0]

        cls_tokens = jnp.repeat(self.cls_token, B, axis=0)

        act_sp_emb = self.species_embed(inputs["act_species"])
        act_t1_emb = self.type_embed(inputs["act_types"][:, :, 0])
        act_t2_emb = self.type_embed(inputs["act_types"][:, :, 1])
        act_cont_emb = self.active_dense(inputs["act_continuous"])
        act_tokens = act_sp_emb + act_t1_emb + act_t2_emb + act_cont_emb

        bench_sp_0 = self.species_embed(inputs["bench_species_p1"])
        bench_cont_0 = self.bench_dense(inputs["bench_cont_p1"])
        bench_tokens_0 = bench_sp_0 + bench_cont_0

        bench_sp_1 = self.species_embed(inputs["bench_species_p2"])
        bench_cont_1 = self.bench_dense(inputs["bench_cont_p2"])
        bench_tokens_1 = bench_sp_1 + bench_cont_1

        move_emb = self.move_embed(inputs["active_moves"])
        move_type_emb = self.type_embed(inputs["move_types"])
        move_cont_emb = self.move_dense(inputs["move_continuous"])
        move_tokens = move_emb + move_type_emb + move_cont_emb

        field_tokens = self.field_dense(inputs["field_features"])[:, None, :]

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
        x = self.tokenize_battlefield(inputs)

        for block in self.encoder_blocks:
            x = block(x, deterministic=deterministic)

        x = self.final_norm(x)
        cls_rep = x[:, 0]

        raw_logits = self.policy_head(cls_rep)

        if valid_mask is not None:
            masked_logits = jnp.where(valid_mask, raw_logits, -1e9)
        else:
            masked_logits = raw_logits

        action_probs = jax.nn.softmax(masked_logits, axis=-1)
        value = jnp.tanh(self.value_head(cls_rep))

        return masked_logits, action_probs, value


def state_to_model_inputs(state: BattleState, perspective_player: int = 0) -> Dict[str, jnp.ndarray]:
    p = perspective_player
    opp = 1 - p

    act_species = jnp.array([state.active_species[p], state.active_species[opp]], dtype=jnp.int32)

    p_t2 = jnp.where(state.active_types[p, 1] >= 0, state.active_types[p, 1], 18)
    opp_t2 = jnp.where(state.active_types[opp, 1] >= 0, state.active_types[opp, 1], 18)
    act_types = jnp.array([
        [state.active_types[p, 0], p_t2],
        [state.active_types[opp, 0], opp_t2],
    ], dtype=jnp.int32)

    def make_active_cont(side: int) -> jnp.ndarray:
        hp = state.active_hp[side:side+1]
        boosts = state.active_boosts[side] / 6.0
        st = state.active_status[side]
        st_onehot = jax.nn.one_hot(st, 7)
        stats = state.active_stats[side] / 300.0
        return jnp.concatenate([hp, boosts, st_onehot, stats])

    act_cont = jnp.stack([make_active_cont(p), make_active_cont(opp)])

    bench_sp_p1 = state.team_species[p, 1:6]
    bench_cont_p1 = jnp.stack([state.team_hp[p, 1:6], state.team_alive[p, 1:6].astype(jnp.float32)], axis=-1)

    bench_sp_p2 = state.team_species[opp, 1:6]
    bench_cont_p2 = jnp.stack([state.team_hp[opp, 1:6], state.team_alive[opp, 1:6].astype(jnp.float32)], axis=-1)

    act_moves = state.active_moves[p]
    m_types = MOVE_TABLE[act_moves, 0]
    m_cats = MOVE_TABLE[act_moves, 1] / 2.0
    m_bps = MOVE_TABLE[act_moves, 2] / 150.0
    m_pris = (MOVE_TABLE[act_moves, 4] + 7.0) / 14.0
    m_pps = state.active_move_pp[p]
    move_cont = jnp.stack([m_cats, m_bps, m_pris, m_pps], axis=-1)

    weather_onehot = jax.nn.one_hot(state.weather, 5)
    terrain_onehot = jax.nn.one_hot(state.terrain, 5)
    hazards_flat = state.hazards.reshape(-1) / 3.0
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


batch_state_to_model_inputs = jax.vmap(state_to_model_inputs, in_axes=(0, None))
