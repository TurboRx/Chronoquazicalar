# Chronos

Vectorized reinforcement learning system for Pokémon Showdown Gen 9 Random Battles.

## Overview

Chronos combines a pure JAX vectorized battle engine with a non-causal Transformer policy/value network and simultaneous-turn regret matching lookahead search.

## Architecture

- **Vectorized Simulator (`engine/`)**: Tensor-native Gen 9 Random Battles engine implemented in JAX (`jax.vmap` and `jax.lax.scan`). Computes turn order resolution, priority brackets, Gen 9 damage formulas, type effectiveness matrices, and field hazards entirely in accelerator memory.
- **Model Architecture (`models/`)**: Non-causal Transformer encoder (~8.5M parameters).
  - Embedding dimension (`d_model`): 256
  - Attention heads: 8
  - Encoder layers: 6
  - Feed-forward dimension (`d_ff`): 1024
  - Dual heads: Policy (9 actions: 4 moves + 5 switches) and scalar Value ($V \in [-1, 1]$).
- **Exact Damage Heuristics (`engine/damage_calc.py`)**: Deterministic min-roll damage calculations and priority checks for branch pruning.
- **Search & Game Theory (`search/`)**:
  - Counterfactual Regret Matching (CFR) to compute mixed Nash equilibria for simultaneous action selection.
  - Polynomial Upper Confidence Trees (pUCT) lookahead search with a strict per-turn execution time budget.
- **Client Interface (`bot_client.py`)**: `poke-env` WebSocket player supporting live ladder play and local battle evaluations.

## Project Structure

```
├── bot_client.py              # Poke-env client and ladder interface
├── engine/
│   ├── battle_state.py        # PyTree state representation
│   ├── damage_calc.py         # Damage calculation and KO heuristics
│   ├── data_extractor.py      # Species, moves, and typechart compiler
│   └── jax_battle_engine.py   # Vectorized simulator
├── models/
│   └── transformer_policy.py  # Transformer policy and value network
├── search/
│   ├── puct_search.py         # pUCT lookahead search
│   └── regret_matching.py     # Matrix game CFR solver
├── training/
│   ├── pretrain_bc.py         # Behavioral cloning pretraining
│   └── selfplay_ppo.py        # PPO self-play training pipeline
└── scripts/
    ├── deploy_hf.py           # Hugging Face deployment utility
    ├── setup_env.py           # Dependency verification
    └── watchdog.py            # Training watchdog daemon
```

## Requirements

- Python 3.10+
- JAX & jaxlib
- Flax
- Optax
- poke-env
- NumPy, SciPy

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Tests
```bash
PYTHONPATH=. python3 engine/test_jax_engine.py
PYTHONPATH=. python3 models/test_transformer.py
PYTHONPATH=. python3 search/test_search.py
```

### Local Evaluation Battles
```bash
python3 bot_client.py --test-local --battles 10
```

### Ladder Play
```bash
python3 bot_client.py --username "<SHOWDOWN_USER>" --password "<SHOWDOWN_PASS>" --ladder
```

### Pretraining (Behavioral Cloning)
```bash
PYTHONPATH=. python3 training/pretrain_bc.py
```

### Self-Play PPO
```bash
PYTHONPATH=. python3 training/selfplay_ppo.py
```
