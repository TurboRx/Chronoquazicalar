"""
evaluation/run_benchmark_eval.py
Head-to-head evaluation harness for Project Chronos against pmariglia's
expert heuristic benchmark engine on local Pokemon Showdown.
"""

import argparse
import asyncio
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_client import ChronosPlayer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("BenchmarkEval")

BOT_REPO_PATH = REPO_ROOT / "pmariglia_bot"


def start_benchmark_bots(
    num_workers: int, search_time_ms: int, battles_per_worker: List[int]
) -> List[subprocess.Popen]:
    """Spawns background processes of pmariglia's heuristic engine."""
    bot_procs = []
    for i in range(num_workers):
        bot_name = f"BenchmarkBot_{i}"
        run_count = battles_per_worker[i]
        cmd = [
            sys.executable,
            "run.py",
            "--websocket-uri",
            "ws://localhost:8000/showdown/websocket",
            "--ps-username",
            bot_name,
            "--bot-mode",
            "accept_challenge",
            "--pokemon-format",
            "gen9randombattle",
            "--search-time-ms",
            str(search_time_ms),
            "--run-count",
            str(run_count),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(BOT_REPO_PATH),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        bot_procs.append(proc)
        logger.info(f"Spawned benchmark bot process '{bot_name}' (PID: {proc.pid}) for {run_count} battles.")
    return bot_procs


async def run_worker_battles(
    worker_idx: int,
    bot_name: str,
    n_battles: int,
    checkpoint_path: Path,
    search_budget: float,
) -> ChronosPlayer:
    """Runs a series of challenge battles against a designated benchmark bot worker."""
    unique_suffix = int(time.time() * 1000) % 100000
    bot_worker_name = f"Chronos_{worker_idx}_{unique_suffix}"
    account_config = AccountConfiguration(bot_worker_name, None)
    player = ChronosPlayer(
        checkpoint_path=checkpoint_path,
        search_time_budget=search_budget,
        account_configuration=account_config,
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )
    logger.info(f"[Worker {worker_idx}] {bot_worker_name} sending {n_battles} challenges to {bot_name}...")
    await player.send_challenges(bot_name, n_challenges=n_battles)
    logger.info(f"[Worker {worker_idx}] Completed {n_battles} battles vs {bot_name}.")
    return player


async def evaluate_checkpoint(
    checkpoint_path: Path,
    num_battles: int = 50,
    num_workers: int = 3,
    search_time_ms: int = 40,
    chronos_search_budget: float = 0.2,
) -> Dict[str, float]:
    """Executes the full parallel evaluation suite against pmariglia's benchmark engine."""
    logger.info(f"=== Starting Evaluation for {checkpoint_path.name} ({num_battles} battles) ===")
    start_time = time.time()

    # Distribute battles across workers
    base_battles = num_battles // num_workers
    remainder = num_battles % num_workers
    battles_per_worker = [base_battles + (1 if i < remainder else 0) for i in range(num_workers)]

    # Start benchmark bot instances
    bot_procs = start_benchmark_bots(num_workers, search_time_ms, battles_per_worker)
    await asyncio.sleep(2.5)  # Wait for bots to connect and log in to Showdown

    try:
        tasks = [
            run_worker_battles(
                worker_idx=i,
                bot_name=f"BenchmarkBot_{i}",
                n_battles=battles_per_worker[i],
                checkpoint_path=checkpoint_path,
                search_budget=chronos_search_budget,
            )
            for i in range(num_workers)
        ]
        players: List[ChronosPlayer] = await asyncio.gather(*tasks)
    finally:
        # Cleanly terminate benchmark bot processes
        for p in bot_procs:
            try:
                p.terminate()
                p.wait(timeout=2)
            except Exception:
                p.kill()

    # Aggregate results across all workers
    total_wins = sum(p.n_won_battles for p in players)
    total_losses = sum(p.n_lost_battles for p in players)
    total_ties = sum(p.n_tied_battles for p in players)
    total_battles = max(total_wins + total_losses + total_ties, 1)
    win_rate = (total_wins / total_battles) * 100.0

    all_turns = []
    all_hp_diffs = []
    for p in players:
        for b in p.battles.values():
            if b.finished:
                all_turns.append(b.turn)
                p1_hp = sum(mon.current_hp_fraction for mon in b.team.values())
                p2_hp = sum(mon.current_hp_fraction for mon in b.opponent_team.values())
                all_hp_diffs.append(p1_hp - p2_hp)

    avg_turns = float(np.mean(all_turns)) if all_turns else 0.0
    avg_hp_diff = float(np.mean(all_hp_diffs)) if all_hp_diffs else 0.0
    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info(f"EVALUATION SUMMARY for {checkpoint_path.name}:")
    logger.info(f"Total Battles: {total_battles} | Wins: {total_wins} | Losses: {total_losses} | Ties: {total_ties}")
    logger.info(f"Win Rate: {win_rate:.2f}% | Avg Turns: {avg_turns:.1f} | HP Diff: {avg_hp_diff:+.2f}")
    logger.info(f"Time Elapsed: {elapsed:.1f}s ({elapsed / total_battles:.1f}s/battle)")
    logger.info("=" * 60)

    return {
        "checkpoint": str(checkpoint_path.name),
        "total_battles": total_battles,
        "wins": total_wins,
        "losses": total_losses,
        "ties": total_ties,
        "win_rate": win_rate,
        "avg_turns": avg_turns,
        "avg_hp_diff": avg_hp_diff,
        "time_elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Run evaluation against pmariglia's benchmark")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_latest.pkl")
    parser.add_argument("--num-battles", type=int, default=50)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--search-time-ms", type=int, default=40)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    asyncio.run(
        evaluate_checkpoint(
            checkpoint_path=ckpt,
            num_battles=args.num_battles,
            num_workers=args.workers,
            search_time_ms=args.search_time_ms,
        )
    )


if __name__ == "__main__":
    main()
