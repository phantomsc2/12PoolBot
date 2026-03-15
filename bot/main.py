import cProfile
import io
import json
import os
import pstats
import random
import sys
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain

from sc2.data import Result

from ares import DEBUG, AresBot
from ares.behaviors.macro import Mining
from leitwerk import Optimizer
from loguru import logger
from sc2.ids.unit_typeid import UnitTypeId

from .combat_predictor_sim import CombatPredictor, CombatPrediction, CombatPredictorParams
from .components.macro import Macro
from .components.micro import Micro
from .components.strategy import Strategy
from .consts import (
    DPS_OVERRIDE,
    EXCLUDE_FROM_COMBAT,
    PROFILING_FILE,
    TAG_ACTION_FAILED,
    TAG_MICRO_THROTTLING,
    UNKNOWN_VERSION,
    VERSION_FILE, PARAMS_FILE,
)
from .tags import Tags


@dataclass(frozen=True)
class BotParams:
    combat_predictor: CombatPredictorParams


class TwelvePoolBot(Strategy, Micro, Macro, AresBot):
    max_micro_actions = 80
    version: str = UNKNOWN_VERSION
    tags: Tags
    optimizer = Optimizer(BotParams)
    params: BotParams

    async def on_start(self) -> None:
        await super().on_start()

        if PARAMS_FILE.exists():
            self.optimizer.load(json.loads(PARAMS_FILE.read_text()))

        context = {
            "enemy_race": self.enemy_race.name
        }
        logger.info(f"{context=}")
        self.params = self.optimizer.ask(context)
        logger.info(f"{self.params=}")

        self.tags = Tags(lambda m: self.chat_send(m, team_only=True))

        if sys.gettrace():
            self.config[DEBUG] = True

        if self.config[DEBUG]:
            # increase number of decimal places
            pstats.f8 = lambda x: "%14.9f" % x  # type: ignore
            # await self.client.debug_create_unit([[UnitTypeId.ZERGLING, 40, self.game_info.map_center, 2]])
            # await self.client.debug_create_unit([[UnitTypeId.ZERGLING, 30, self.game_info.map_center, 1]])

        if os.path.exists(VERSION_FILE):
            with open(VERSION_FILE) as f:
                self.version = f.read()

        await self.tags.add_tag(f"version_{self.version}")

    async def on_step(self, iteration: int) -> None:
        await super().on_step(iteration)

        profiler: cProfile.Profile | None = None
        if self.config[DEBUG] and (iteration % 30) == 10:
            profiler = cProfile.Profile()

        if profiler:
            profiler.enable()
        strategy = self.decide_strategy()

        
        units = self.all_own_units.exclude_type(EXCLUDE_FROM_COMBAT)
        enemy_units = self.all_enemy_units.exclude_type(EXCLUDE_FROM_COMBAT)
        predictor = CombatPredictor(self, units, enemy_units, self.params.combat_predictor)

        if strategy.build_unit not in {UnitTypeId.ZERGLING, UnitTypeId.DRONE}:
            await self.tags.add_tag(f"macro_{strategy.build_unit.name}")
        if self.mediator.get_own_army_dict[UnitTypeId.ROACH]:
            await self.tags.add_tag("macro_ROACH")

        pathing = self.mediator.get_ground_grid.astype(float)
        macro_actions = list(self.macro(strategy.build_unit))
        micro_actions = list(self.micro(predictor, pathing, self.supply_used))

        # avoid APM bug
        if self.max_micro_actions < len(micro_actions):
            await self.tags.add_tag(TAG_MICRO_THROTTLING)
            logger.info(f"Limiting micro actions: {len(micro_actions)} => {self.max_micro_actions}")
            random.shuffle(micro_actions)
            micro_actions = micro_actions[: self.max_micro_actions]

        if profiler:
            profiler.disable()
            stats_io = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_io).sort_stats(pstats.SortKey.TIME).print_stats(24)
            logger.info(stats_io.getvalue())
            stats.dump_stats(PROFILING_FILE)

        actions = chain(macro_actions, micro_actions)
        for action in actions:
            success = await action.execute(self)
            if not success:
                await self.tags.add_tag(TAG_ACTION_FAILED)
                if self.config[DEBUG]:
                    raise Exception(f"Action failed: {action}")
                else:
                    logger.warning(f"Action failed: {action}")

        self.register_behavior(Mining(workers_per_gas=strategy.vespene_target))

    @lru_cache(maxsize=None)
    def dps_fast(self, unit: UnitTypeId) -> float:
        if dps := DPS_OVERRIDE.get(unit):
            return dps
        elif units := self.all_units(unit):
            return max(units[0].ground_dps, units[0].air_dps)
        else:
            return 0.0

    async def on_end(self, game_result: Result) -> None:
        await super().on_end(game_result)

        PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
