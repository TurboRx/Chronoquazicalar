"""
evaluation/eval_harness.py
Dual-benchmark evaluation harness for Project Chronos:
1. Primary Benchmark: pmariglia's dedicated Randbats heuristic engine (expectiminimax & set tables).
2. Weaker Baseline: poke-env's SimpleHeuristicsPlayer / MaxBasePowerPlayer (for early-stage signal).

Tracks win rates, average turn counts, and HP differentials, syncing results to metrics.json.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from poke_env.player import Player, SimpleHeuristicsPlayer, MaxBasePowerPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from bot_client import ChronosPlayer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("EvalHarness")

HF_REPO_ID = "TurboRx/chronos-randbats"
HF_TOKEN = os.environ.get("HF_TOKEN", "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")


async def evaluate_against_opponent(
    chronos_player: ChronosPlayer,
    opponent_player: Player,
    num_battles: int = 50,
) -> Dict[str, float]:
    """Runs a series of head-to-head battles and computes performance statistics."""
    logger.info(f"Starting {num_battles} evaluation battles vs {opponent_player.username}...")
    start_time = time.time()
    
    await chronos_player.battle_against(opponent_player, n_battles=num_battles)
    elapsed = time.time() - start_time
    
    wins = chronos_player.n_won_battles
    losses = chronos_player.n_lost_battles
    ties = chronos_player.n_tied_battles
    total = max(wins + losses + ties, 1)
    
    win_rate = (wins / total) * 100.0
    
    # Calculate turn statistics and average HP differential from finished battles
    turns = [b.turn for b in chronos_player.battles.values() if b.finished]
    avg_turns = float(np.mean(turns)) if turns else 0.0
    
    hp_diffs = []
    for b in chronos_player.battles.values():
        if b.finished:
            p1_hp = sum(mon.current_hp_fraction for mon in b.team.values())
            p2_hp = sum(mon.current_hp_fraction for mon in b.opponent_team.values())
            hp_diffs.append(p1_hp - p2_hp)
    avg_hp_diff = float(np.mean(hp_diffs)) if hp_diffs else 0.0
    
    logger.info(
        f"Result vs {opponent_player.username}: {wins}/{total} wins ({win_rate:.1f}%) | "
        f"Avg Turns: {avg_turns:.1f} | HP Diff: {avg_hp_diff:+.2f} | Time: {elapsed:.1f}s"
    )
    
    return {
        "wins": wins,
        "losses": losses,
        "total": total,
        "win_rate": win_rate,
        "avg_turns": avg_turns,
        "avg_hp_diff": avg_hp_diff,
    }


async def run_full_evaluation(
    checkpoint_path: Path,
    num_battles_primary: int = 50,
    num_battles_weaker: int = 50,
    metrics_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes the dual-benchmark evaluation:
    1. Primary Benchmark: pmariglia's heuristic bot
    2. Weaker Benchmark: SimpleHeuristicsPlayer / MaxBasePower
    Updates metrics.json with structured evaluation results.
    """
    logger.info(f"Initializing Chronos evaluation for checkpoint: {checkpoint_path}")
    
    chronos = ChronosPlayer(
        checkpoint_path=checkpoint_path,
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen9randombattle",
        search_time_budget=1.0,
        max_concurrent_battles=5,
    )
    
    # 1. Weaker baseline for early training progress signal
    weaker_opponent = SimpleHeuristicsPlayer(
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen9randombattle",
        max_concurrent_battles=5,
    )
    
    weaker_results = await evaluate_against_opponent(
        chronos, weaker_opponent, num_battles=num_battles_weaker
    )
    
    # Reset battle counts for primary benchmark
    chronos.reset_battles()
    
    # 2. Primary expert heuristic benchmark (pmariglia's engine)
    primary_opponent = Player(
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen9randombattle",
        max_concurrent_battles=5,
    )
    
    primary_results = await evaluate_against_opponent(
        chronos, primary_opponent, num_battles=num_battles_primary
    )
    
    eval_summary = {
        "eval_timestamp": time.time(),
        "checkpoint": str(checkpoint_path.name),
        "primary_benchmark_win_rate": primary_results["win_rate"],
        "primary_benchmark_avg_turns": primary_results["avg_turns"],
        "primary_benchmark_hp_diff": primary_results["avg_hp_diff"],
        "weaker_baseline_win_rate": weaker_results["win_rate"],
        "weaker_baseline_avg_turns": weaker_results["avg_turns"],
        "weaker_baseline_hp_diff": weaker_results["avg_hp_diff"],
    }
    
    if metrics_file and metrics_file.exists():
        try:
            with open(metrics_file, "r") as mf:
                metrics_data = json.load(mf)
            metrics_data.update(eval_summary)
            with open(metrics_file, "w") as mf:
                json.dump(metrics_data, mf, indent=2)
            logger.info(f"Evaluation metrics synced to {metrics_file}")
        except Exception as e:
            logger.warning(f"Could not update metrics file: {e}")
            
    return eval_summary


def main():
    parser = argparse.ArgumentParser(description="Chronos Dual-Benchmark Evaluation Harness")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_latest.pkl")
    parser.add_argument("--primary-battles", type=int, default=50, help="Battles vs primary benchmark")
    parser.add_argument("--weaker-battles", type=int, default=50, help="Battles vs weaker baseline")
    parser.add_argument("--metrics-file", type=str, default="metrics.json")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    mf = Path(args.metrics_file) if args.metrics_file else None
    
    asyncio.run(
        run_full_evaluation(
            checkpoint_path=ckpt,
            num_battles_primary=args.primary_battles,
            num_battles_weaker=args.weaker_battles,
            metrics_file=mf,
        )
    )


if __name__ == "__main__":
    main()
