#!/usr/bin/env python3
"""
scripts/benchmark_showdown.py
Automated benchmarking harness for Project Chronos against baseline players
using the official local Pokémon Showdown server and simulator.
"""

import argparse
import asyncio
import subprocess
import time
from pathlib import Path

from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from bot_client import ChronosPlayer


def is_server_running(port: int = 8000) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_local_server(showdown_dir: Path, port: int = 8000) -> subprocess.Popen:
    print(f"Starting local Pokémon Showdown server on port {port}...")
    cmd = ["node", "pokemon-showdown", "start", "--no-security", f"--port={port}"]
    proc = subprocess.Popen(
        cmd,
        cwd=str(showdown_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for server to accept connections
    for _ in range(30):
        if is_server_running(port):
            print("Local Pokémon Showdown server is ready!")
            return proc
        time.sleep(0.5)
    raise TimeoutError(
        "Failed to start local Pokémon Showdown server within 15 seconds."
    )


async def benchmark(
    num_battles: int = 10,
    opponent_type: str = "heuristic",
    checkpoint_path: str = None,
    time_budget: float = 0.5,
):
    print("\n==========================================")
    print("PROJECT CHRONOS - SHOWDOWN BENCHMARK")
    print(
        f"Battles: {num_battles} | Opponent: {opponent_type} | Time Budget: {time_budget}s"
    )
    print("==========================================\n")

    ckpt = Path(checkpoint_path) if checkpoint_path else None
    chronos = ChronosPlayer(
        checkpoint_path=ckpt,
        search_time_budget=time_budget,
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )

    if opponent_type.lower() == "random":
        opp = RandomPlayer(
            server_configuration=LocalhostServerConfiguration,
            battle_format="gen9randombattle",
            max_concurrent_battles=1,
        )
    else:
        opp = SimpleHeuristicsPlayer(
            server_configuration=LocalhostServerConfiguration,
            battle_format="gen9randombattle",
            max_concurrent_battles=1,
        )

    start_time = time.time()
    await chronos.battle_against(opp, n_battles=num_battles)
    elapsed = time.time() - start_time

    wins = chronos.n_won_battles
    losses = opp.n_won_battles
    winrate = (wins / num_battles) * 100.0 if num_battles > 0 else 0.0

    print("\n==========================================")
    print("BENCHMARK RESULTS")
    print("==========================================")
    print(f"Chronos Wins:   {wins} / {num_battles} ({winrate:.1f}%)")
    print(f"Opponent Wins:  {losses} / {num_battles}")
    print(f"Total Time:     {elapsed:.1f}s ({elapsed / num_battles:.2f}s per battle)")
    print("==========================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Chronos against baseline players on local Showdown."
    )
    parser.add_argument(
        "--battles", type=int, default=5, help="Number of benchmark battles"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        choices=["heuristic", "random"],
        default="heuristic",
        help="Opponent type",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model weights checkpoint"
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=0.5,
        help="Search time budget per turn in seconds",
    )
    args = parser.parse_args()

    showdown_dir = Path(__file__).resolve().parent.parent / "pokemon-showdown"
    proc = None
    if not is_server_running(8000):
        proc = start_local_server(showdown_dir, 8000)

    try:
        asyncio.run(
            benchmark(
                num_battles=args.battles,
                opponent_type=args.opponent,
                checkpoint_path=args.checkpoint,
                time_budget=args.time_budget,
            )
        )
    finally:
        if proc:
            print("Stopping local Pokémon Showdown server...")
            proc.terminate()
            proc.wait()


if __name__ == "__main__":
    main()
