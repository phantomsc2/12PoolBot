from bot import TwelvePoolBot
from sc2 import maps
from sc2.data import AIBuild, Difficulty, Race, Result
from sc2.main import run_game
from sc2.player import Bot, Computer

if __name__ == "__main__":
    result = run_game(
        maps.get("Equilibrium513AIE"),
        [
            Bot(Race.Zerg, TwelvePoolBot(), "12PoolBot"),
            Computer(Race.Random, Difficulty.VeryEasy, ai_build=AIBuild.Macro),
        ],
        realtime=False,
        save_replay_as="replay.SC2Replay",
    )
    assert result == Result.Victory
