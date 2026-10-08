# Chronoquazicalar

High-performance reinforcement learning system and game-theoretic search engine for Pokémon Showdown Gen 9 Random Battles.

---

## Overview

Chronoquazicalar integrates a tensor-native JAX battle simulator, a 9.65M-parameter non-causal Transformer policy/value network, simultaneous-move Counterfactual Regret Minimization (CFR) with polynomial Upper Confidence Trees (pUCT), and Bayesian set inference over official Pokémon Showdown Gen 9 distributions.

Pretrained model checkpoints and training metrics are hosted publicly on Hugging Face: [TurboRx/Chronoquazicalar](https://huggingface.co/TurboRx/Chronoquazicalar).

---

## System Architecture

### 1. Vectorized Simulation Engine (`engine/`)
* **Hardware-Accelerated Physics:** Pure JAX battle simulator utilizing `jax.vmap` and `jax.lax.scan` for massive parallel rollouts (4,096+ environments, >250 battles/sec on single GPU/TPU).
* **Deterministic Mechanics:** Exact Gen 9 damage calculation formulas, speed tier resolving, priority brackets, stat stage multipliers, status conditions, entry hazards (Stealth Rock, Spikes, Toxic Spikes, Sticky Web), weather, and terrain.
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
  * **Opponent Prediction Head:** Predicts the opponent's simultaneous action to model counter-play.

### 3. Game-Theoretic Search Engine (`search/`)
* **Simultaneous-Move pUCT:** Looks ahead 3–6 turns, evaluating candidate branches against the opponent's counter-play using neural priors and exact damage rollouts.
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
│   ├── damage_calc.py            # Exact damage formulas and KO heuristics
│   ├── data_extractor.py         # Mechanics tables and mappings compiler
│   ├── jax_battle_engine.py      # Vectorized JAX battle simulator
│   └── randbats_knowledge.py     # Gen 9 set inference engine (509 species)
├── models/
│   ├── test_transformer.py       # Parameter count & latency verification
│   └── transformer_policy.py     # 8.33M Transformer policy/value architecture
├── search/
│   ├── puct_search.py            # Simultaneous-move pUCT tree search
│   ├── regret_matching.py        # Matrix game CFR solver
│   └── test_search.py            # Search & Nash convergence tests
├── training/
│   ├── pretrain_bc.py            # Behavioral cloning pretraining pipeline
│   └── selfplay_ppo.py           # Vectorized PPO self-play pipeline
├── scripts/
│   ├── benchmark_showdown.py     # Automated local battle benchmarking harness
│   ├── cloud_watchdog.py         # Autonomous cloud training recovery daemon
│   ├── deploy_hf.py              # Checkpoint deployment utility
│   ├── setup_env.py              # Environment and dependency verification
│   └── update_showdown_data.py   # Showdown upstream dataset synchronization
└── .github/workflows/
    ├── update-data.yml           # Daily automated dataset update workflow
    └── watchdog.yml              # 24/7 cloud training watchdog workflow
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

Run the full unit test suite:
```bash
python3 -m unittest discover -s engine -p "test_*.py"
python3 -m unittest discover -s models -p "test_*.py"
python3 -m unittest discover -s search -p "test_*.py"
```

---

## Usage

### 1. Local Battle Benchmarking
Benchmark Chronos against baseline players on a local Pokémon Showdown server:
```bash
python3 scripts/benchmark_showdown.py --battles 10 --opponent heuristic --time-budget 1.0
```

### 2. Live Ladder Play
Connect Chronos to the official Pokémon Showdown ladder:
```bash
python3 bot_client.py --username "<SHOWDOWN_USER>" --password "<SHOWDOWN_PASS>" --ladder
```

### 3. Upstream Data Synchronization
Manually synchronize Gen 9 Random Battle datasets and mechanics tables from upstream Pokémon Showdown:
```bash
python3 scripts/update_showdown_data.py --showdown-dir ./pokemon-showdown
```
*(This is also automated to run daily via GitHub Actions).*

### 4. Training Pipelines
* **Behavioral Cloning Pretraining:**
  ```bash
  python3 training/pretrain_bc.py
  ```
* **PPO Self-Play Training:**
  ```bash
  python3 training/selfplay_ppo.py
  ```

---

## License

Private and proprietary. All rights reserved.
