"""
evaluation/play_against_user.py
Interactive Showdown battle runner:
Runs ChronosBot on the local Showdown server and continuously accepts
challenges from the user (TurboRx) or any challenger in gen9randombattle.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poke_env.ps_client.account_configuration import AccountConfiguration  # noqa: E402
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration  # noqa: E402

from bot_client import ChronosPlayer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("UserBattle")


async def run_bot():
    checkpoint_file = REPO_ROOT / "checkpoints" / "checkpoint_latest.pkl"
    account = AccountConfiguration("ChronosBot", None)
    player = ChronosPlayer(
        checkpoint_path=checkpoint_file,
        search_time_budget=0.5,
        account_configuration=account,
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )

    logger.info("=" * 60)
    logger.info("ChronosBot is ONLINE and waiting for your challenge!")
    logger.info("Username: ChronosBot")
    logger.info("Format: [Gen 9] Random Battle")
    logger.info("=" * 60)

    # Accept up to 50 challenges consecutively
    await player.accept_challenges(opponent=None, n_challenges=50)


if __name__ == "__main__":
    asyncio.run(run_bot())
