# Chronoquazicalar

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Chronoquazicalar-blue)](https://huggingface.co/TurboRx/Chronoquazicalar)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX Hardware-Accelerated](https://img.shields.io/badge/JAX-Hardware--Accelerated-green.svg)](https://github.com/google/jax)

High-performance reinforcement learning system and game-theoretic search engine for **Pokémon Showdown Gen 9 Random Battles**.

---

## Overview

**Chronoquazicalar** integrates a tensor-native JAX battle simulator, a 9.65M-parameter non-causal Transformer policy/value network, simultaneous-move Counterfactual Regret Minimization (CFR) with polynomial Upper Confidence Trees (pUCT), and Bayesian set inference over official Pokémon Showdown Gen 9 distributions.

Pretrained model checkpoints and live training metrics are hosted publicly on Hugging Face: [**TurboRx/Chronoquazicalar**](https://huggingface.co/TurboRx/Chronoquazicalar).

---

## Key Highlights

- **Tensor-Native JAX Simulator:** Simulates thousands of parallel battle environments (`jax.vmap` & `jax.lax.scan`) in accelerator memory, reaching **>4,400 turns/second** on TPU v3-8 and scaling to 140M+ battle turns.
- **Dual & Triple-Head Transformer (~9.65M Parameters):** Factored continuous state, active/bench move features, field conditions, policy head (4 moves + 5 switches), value head ($V \in [-1, 1]$), and an auxiliary opponent action prediction head for counter-play modeling.
- **Vectorized Sparring Partner:** Built-in pure JAX vectorized implementation of the expert heuristic benchmark opponent capable of evaluating **600,000+ decisions/second** across parallel environments.
- **Simultaneous-Move Game-Theoretic Search:** Combines multi-scenario belief determinization with simultaneous-move CFR matrix regret-matching to approximate unexploitable **Nash Equilibrium** mixed strategies.
- **Bayesian Set Inference (509 Species):** Exact Randbats knowledge base containing all Gen 9 species, level scalings, movepools, ability pools, and Tera distributions extracted directly from Pokémon Showdown.
- **Tactical Terastallization & Immunity Resolution:** Dynamic defensive weakness-inverting Tera, offensive STAB KO thresholding, and ability-aware immunity checks covering revealed and unrevealed candidate pools (Levitate, Earth Eater, Flash Fire, Water Absorb, Volt Absorb, Sap Sipper).

---

## System Architecture

### 1. Vectorized Simulation Engine (`engine/`)
* **Hardware-Accelerated Physics:** Pure JAX battle simulator running rollouts across hundreds of parallel environments directly on GPU/TPU.
* **Deterministic Mechanics:** Exact Gen 9 damage calculation formulas, speed tier resolving, priority brackets, stat stage multipliers, status conditions, entry hazards (Stealth Rock, Spikes, Toxic Spikes, Sticky Web), weather, and terrain.
* **Heuristic Sparring Partner (`engine/heuristic_bot_jax.py`):** Fully vectorized JAX opponent enabling hybrid self-play (e.g. 40% heuristic opponent sparring + 15% league pool + 45% self-play).
* **Full Team State Tracking:** Immutable PyTree state (`BattleState`) maintaining species, active/bench movepools, PP, HP fractions, and field conditions across both sides.

### 2. Neural Policy & Value Network (`models/`)
* **Non-Causal Transformer Encoder (~9.65M Parameters):**
  * Embedding Dimension ($d_{\text{model}}$): 256
  * Multi-Head Attention: 8 heads
  * Encoder Depth: 6 layers
  * Feed-Forward Dimension ($d_{\text{ff}}$): 1024
  * Positional & Type Embeddings: Factored continuous active state, bench status, move properties, and battlefield features.
* **Triple-Head Architecture:**
  * **Policy Head:** 9 discrete actions (4 active moves + 5 bench switches) with legal action masking.
  * **Value Head:** Scalar game-state valuation ($V \in [-1, 1]$).
  * **Opponent Prediction Head:** Auxiliary classification head predicting the opponent's simultaneous action to model counter-play.

### 3. Game-Theoretic Search Engine (`search/`)
* **Simultaneous-Move pUCT:** Looks ahead 3–6 turns, evaluating candidate branches against opponent counter-play using neural priors and exact damage rollouts.
* **CFR Regret Matching:** Solves the simultaneous turn matrix at the root node for an approximate **Nash Equilibrium**, yielding unexploitable mixed strategies (e.g. mixed attack/switch frequencies).
* **Multi-Scenario Determinization:** When an opposing Pokémon has unrevealed moves or multiple viable set archetypes, the engine samples candidate roles from the knowledge base, searches each scenario, and aggregates the resulting Nash equilibrium distributions.
* **Deterministic KO Pruning:** Heuristic checks identify outspeeding guaranteed lethal attacks to eliminate unnecessary search overhead.

### 4. Bayesian Set Inference & Meta Tracking (`engine/randbats_knowledge.py`)
* **Ground-Truth Distributions:** Complete database of all 509 Gen 9 Random Battle species extracted directly from Pokémon Showdown (`sets.json`).
* **Belief State Resolution:** Dynamically filters candidate roles, movepools, abilities, and Tera types as opponent moves are revealed.
* **Exact Level Scaling:** Provides exact Randbats levels for every Pokémon to ensure zero-error damage calculation thresholds.

### 5. Client & Decision Logic (`bot_client.py`)
* **Poke-Env WebSocket Integration:** Asynchronous client for official Pokémon Showdown servers and local headless instances.
* **Tactical Terastallization:**
  * *Defensive Counter-Tera:* Flips unfavorable matchups when facing super-effective lethal attacks ($\ge 2.0\times$) into resistances ($\le 0.5\times$) or immunities ($0.0\times$).
  * *Offensive STAB Boost:* Triggers when Tera STAB turns a 2HKO into a guaranteed 1HKO against key opposing threats.
  * *Critical Preservation:* Low-HP survival logic on active setup sweepers.

---

## Project Structure

```
├── bot_client.py                 # Poke-env client and ladder interface
├── engine/
│   ├── battle_state.py           # Immutable PyTree state dataclass
│   ├── belief_sampler.py         # Monte Carlo belief-state determinization
│   ├── damage_calc.py            # Exact damage formulas and KO heuristics
│   ├── data_extractor.py         # Mechanics tables and mappings compiler
│   ├── heuristic_bot_jax.py      # Vectorized JAX benchmark sparring opponent
│   ├── jax_battle_engine.py      # Vectorized JAX battle simulator
│   └── randbats_knowledge.py     # Gen 9 set inference engine (509 species)
├── evaluation/
│   ├── eval_harness.py           # Evaluation harness against baseline players
│   ├── play_against_user.py      # Interactive challenge server for human play
│   ├── run_benchmark_eval.py     # Multi-worker evaluation against heuristic benchmark
│   └── run_continuous_eval.py    # Background continuous evaluation supervisor
├── models/
│   ├── test_edge_cases.py        # Model masking and extreme state tests
│   ├── test_transformer.py       # Parameter count & latency verification
│   └── transformer_policy.py     # 9.65M Transformer policy/value architecture
├── search/
│   ├── puct_search.py            # Simultaneous-move pUCT tree search
│   ├── regret_matching.py        # Matrix game CFR solver
│   └── test_search.py            # Search & Nash convergence tests
├── training/
│   ├── pretrain_bc.py            # Behavioral cloning pretraining pipeline
│   └── selfplay_ppo.py           # Vectorized PPO self-play pipeline (TPU/GPU)
├── scripts/
│   ├── benchmark_showdown.py     # Local battle benchmarking script
│   ├── check_progress.py         # Training metrics inspection utility
│   ├── deploy_hf.py              # Checkpoint deployment utility
│   ├── setup_env.py              # Environment and dependency verification
│   ├── update_showdown_data.py   # Upstream Pokémon Showdown data updater
│   └── watchdog.py               # Autonomous training watchdog daemon
└── .github/workflows/
    ├── lint.yml                  # Code quality & unit testing CI
    └── update-data.yml           # Daily upstream dataset sync workflow
```

---

## Installation

### Requirements
* Python `>= 3.10`
* JAX & jaxlib
* Flax & Optax
* poke-env
* NumPy, SciPy

```bash
git clone https://github.com/TurboRx/Chronoquazicalar.git
cd Chronoquazicalar
pip install -r requirements.txt
```

---

## Verification & Testing

Run the unit test suites:
```bash
# Engine core and edge cases
python3 -m unittest discover -s engine -p "test_*.py"

# Transformer policy and masking tests
python3 -m unittest discover -s models -p "test_*.py"

# Simultaneous pUCT and CFR search tests
python3 -m unittest discover -s search -p "test_*.py"
```

---

## Usage

### 1. Play Interactively Against the Bot
Launch an interactive local challenge bot. Open your browser on a local Pokémon Showdown server and challenge `ChronosBot` to a `[Gen 9] Random Battle`:
```bash
python3 evaluation/play_against_user.py
```

### 2. Multi-Worker Benchmark Evaluation
Evaluate the latest trained model against the expert heuristic benchmark across multiple workers:
```bash
python3 evaluation/run_benchmark_eval.py --checkpoint checkpoints/checkpoint_latest.pkl --num-battles 50 --workers 3
```

### 3. Local Battle Benchmarking Script
Run local automated benchmark matches with customizable search time budgets:
```bash
python3 scripts/benchmark_showdown.py --battles 10 --opponent heuristic --time-budget 0.5
```

### 4. Live Pokémon Showdown Ladder Play
Connect Chronoquazicalar to the official public Pokémon Showdown ladder:
```bash
python3 bot_client.py --username "<SHOWDOWN_USER>" --password "<SHOWDOWN_PASS>" --ladder --battles 10
```

### 5. Reinforcement Learning Training (PPO Self-Play)
Run high-throughput vectorized PPO self-play training on JAX (automatically detects TPU/GPU):
```bash
python3 training/selfplay_ppo.py
```

### 6. Upstream Data Synchronization
Manually synchronize Gen 9 Random Battle datasets and mechanics tables from upstream Pokémon Showdown:
```bash
python3 scripts/update_showdown_data.py --showdown-dir ./pokemon-showdown
```
*(Automated daily via GitHub Actions).*

---

## License

This project is open-source software licensed under the [MIT License](LICENSE).
