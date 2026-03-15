import json
import os
import random
import sys
from dataclasses import dataclass
from itertools import chain

from ares import DEBUG, AresBot
from ares.behaviors.macro import Mining
from leitwerk import Optimizer
from loguru import logger
from sc2.data import Result
from sc2.ids.unit_typeid import UnitTypeId

from .combat_predictor import CombatPredictor, CombatPredictorParams
from .components.macro import Macro
from .components.micro import Micro
from .components.strategy import Strategy
from .consts import (
    EXCLUDE_FROM_COMBAT,
    PARAMS_FILE,
    TAG_ACTION_FAILED,
    TAG_MICRO_THROTTLING,
    UNKNOWN_VERSION,
    VERSION_FILE,
)


@dataclass(frozen=True)
class BotParams:
    combat_predictor: CombatPredictorParams


class TwelvePoolBot(Strategy, Micro, Macro, AresBot):
    def __init__(self) -> None:
        super().__init__()
        self.max_micro_actions = 80
        self.version: str = UNKNOWN_VERSION
        self.optimizer = Optimizer(BotParams)
        self._tags: set[str] = set()

    async def on_start(self) -> None:
        await super().on_start()

        if PARAMS_FILE.exists():
            schema_diff = self.optimizer.load(json.loads(PARAMS_FILE.read_text()))
            logger.info(f"{schema_diff=}")

        context = {"enemy_race": self.enemy_race.name}
        logger.info(f"{context=}")
        self.params = self.optimizer.ask(context)
        logger.info(f"{self.params=}")

        if sys.gettrace():
            self.config[DEBUG] = True

        if os.path.exists(VERSION_FILE):
            with open(VERSION_FILE) as f:
                self.version = f.read()

        await self.add_tag(f"version_{self.version}")

    async def on_step(self, iteration: int) -> None:
        await super().on_step(iteration)

        strategy = self.decide_strategy()

        units = self.all_own_units.exclude_type(EXCLUDE_FROM_COMBAT)
        enemy_units = self.all_enemy_units.exclude_type(EXCLUDE_FROM_COMBAT)
        predictor = CombatPredictor(self, units, enemy_units, self.params.combat_predictor)

        if strategy.build_unit not in {UnitTypeId.ZERGLING, UnitTypeId.DRONE}:
            await self.add_tag(f"macro_{strategy.build_unit.name}")
        if self.mediator.get_own_army_dict[UnitTypeId.ROACH]:
            await self.add_tag("macro_ROACH")

        pathing = self.mediator.get_ground_grid.astype(float)
        macro_actions = list(self.macro(strategy.build_unit))
        micro_actions = list(self.micro(predictor, pathing, self.supply_used))

        # avoid APM limit
        if self.max_micro_actions < len(micro_actions):
            await self.add_tag(TAG_MICRO_THROTTLING)
            logger.info(f"Limiting micro actions: {len(micro_actions)} => {self.max_micro_actions}")
            random.shuffle(micro_actions)
            micro_actions = micro_actions[: self.max_micro_actions]

        actions = chain(macro_actions, micro_actions)
        for action in actions:
            success = await action.execute(self)
            if not success:
                await self.add_tag(TAG_ACTION_FAILED)
                if self.config[DEBUG]:
                    raise Exception(f"Action failed: {action}")
                else:
                    logger.warning(f"Action failed: {action}")

        self.register_behavior(Mining(workers_per_gas=strategy.vespene_target))

    async def on_end(self, game_result: Result) -> None:
        await super().on_end(game_result)
        outcome = {
            Result.Victory: 1.0,
            Result.Tie: 0.5,
            Result.Defeat: 0.0,
        }[game_result]
        logger.info(f"{outcome=}")
        report = self.optimizer.tell(outcome)
        logger.info(f"{report=}")
        PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        PARAMS_FILE.write_text(json.dumps(self.optimizer.save(), indent=2))

    async def add_tag(self, tag: str) -> bool:
        if tag in self._tags:
            return False
        await self.chat_send(f"Tag:{tag}", team_only=True)
        self._tags.add(tag)
        return True
