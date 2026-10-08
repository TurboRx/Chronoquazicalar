"""
evaluation/run_continuous_eval.py
Runs a continuous live stream of head-to-head battles between Project Chronos
and pmariglia's expert heuristic benchmark engine on local Pokemon Showdown.
"""

import asyncio
import logging
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poke_env.ps_client.account_configuration import AccountConfiguration  # noqa: E402
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration  # noqa: E402

from bot_client import ChronosPlayer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s UTC] %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/eval_live.log", mode="a"),
    ],
)
logger = logging.getLogger("LiveEval")

BOT_REPO_PATH = REPO_ROOT / "pmariglia_bot"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "checkpoint_latest.pkl"


def start_benchmark_bot(run_count: int = 100, search_time_ms: int = 40) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "run.py",
        "--websocket-uri",
        "ws://localhost:8000/showdown/websocket",
        "--ps-username",
        "BenchmarkBot_0",
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
    return proc


async def run_live_battles(total_target: int = 100):
    logger.info("=" * 60)
    logger.info(f"Starting Live Head-to-Head Evaluation ({total_target} Battles)")
    logger.info(f"Model Checkpoint: {CHECKPOINT_PATH.name}")
    logger.info("=" * 60)

    # Spawn benchmark bot
    bot_proc = start_benchmark_bot(run_count=total_target)
    await asyncio.sleep(2.5)

    wins = 0
    losses = 0
    ties = 0

    try:
        unique_id = int(time.time()) % 10000
        account = AccountConfiguration(f"Chronos_{unique_id}", None)
        player = ChronosPlayer(
            checkpoint_path=CHECKPOINT_PATH,
            search_time_budget=0.6,
            account_configuration=account,
            server_configuration=LocalhostServerConfiguration,
            max_concurrent_battles=1,
        )

        prev_wins = 0
        for b_idx in range(1, total_target + 1):
            logger.info(f"\n>>> [Battle {b_idx}/{total_target}] Chronos challenging BenchmarkBot_0...")
            battle_start = time.time()
            await player.send_challenges("BenchmarkBot_0", n_challenges=1)
            b_dur = time.time() - battle_start

            # Tally stats
            wins = player.n_won_battles
            losses = player.n_lost_battles
            ties = player.n_tied_battles
            played = wins + losses + ties
            win_pct = (wins / played * 100.0) if played > 0 else 0.0
            won_this = wins > prev_wins
            prev_wins = wins

            logger.info(
                f"[Battle {b_idx} Done] Result: {'WIN' if won_this else 'LOSS'} | "
                f"Tally: {wins}W - {losses}L - {ties}T ({win_pct:.1f}% Win Rate) | "
                f"Duration: {b_dur:.1f}s"
            )
            await asyncio.sleep(1.0)

    except Exception as e:
        logger.error(f"Error in battle loop: {e}", exc_info=True)
    finally:
        try:
            bot_proc.terminate()
            bot_proc.wait(timeout=2)
        except Exception:
            bot_proc.kill()

    logger.info("=" * 60)
    logger.info(f"Evaluation Complete! Final Score: {wins} Wins / {losses} Losses / {ties} Ties")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_live_battles(total_target=50))
