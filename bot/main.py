import random
from dataclasses import dataclass

import numpy as np
from ares import AresBot
from ares.behaviors.macro import (
    AutoSupply,
    BuildWorkers,
    ExpansionController,
    GasBuildingController,
    MacroPlan,
    Mining,
    SpawnController,
    UpgradeController,
)
from leitwerk import OptimizerSession
from loguru import logger
from sc2.data import Result
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId

from bot.combat_predictor import CombatPredictor, CombatPredictorParams
from bot.components.micro import Micro, MicroParams
from bot.components.strategy import Strategy, StrategyDecision
from bot.consts import (
    EXCLUDE_FROM_COMBAT,
    MAX_MICRO_ACTIONS,
    PARAMS_FILE,
    VERSION_FILE,
)


@dataclass(frozen=True)
class BotParams:
    combat_predictor: CombatPredictorParams
    micro: MicroParams


class TwelvePoolBot(Strategy, Micro, AresBot):
    def __init__(self) -> None:
        super().__init__()
        self.optimizer = OptimizerSession(PARAMS_FILE, BotParams)
        self.params = self.optimizer.mean

    async def on_start(self) -> None:
        await super().on_start()

        if self.optimizer.schema_diff:
            logger.info(f"{self.optimizer.schema_diff}")

        context = {"enemy_race": self.enemy_race.name}
        logger.info(f"{context=}")
        self.params = self.optimizer.ask(context)
        logger.info(f"{self.params=}")

        if VERSION_FILE.exists():
            version = VERSION_FILE.read_text()
            logger.info(f"{version=}")

    async def on_step(self, iteration: int) -> None:
        await super().on_step(iteration)

        strategy = self.decide_strategy()
        units = self.all_own_units.exclude_type(EXCLUDE_FROM_COMBAT)
        enemy_units = self.all_enemy_units.exclude_type(EXCLUDE_FROM_COMBAT)
        predictor = CombatPredictor(self, units, enemy_units, self.params.combat_predictor)
        micro_actions = list(self.micro(predictor, self.params.micro))

        # avoid APM limit
        max_micro_actions = MAX_MICRO_ACTIONS
        if max_micro_actions < len(micro_actions):
            logger.info(f"Limiting micro actions: {len(micro_actions)} => {max_micro_actions}")
            random.shuffle(micro_actions)
            micro_actions = micro_actions[:max_micro_actions]

        for action in micro_actions:
            await action.execute(self)

        self.register_behavior(Mining(workers_per_gas=strategy.vespene_target))

        if self.build_order_runner.build_completed:
            macro = self._macro(strategy)
            self.register_behavior(macro)

    async def on_end(self, game_result: Result) -> None:
        await super().on_end(game_result)

        if self.params:
            outcome = {
                Result.Victory: 1.0,
                Result.Tie: 0.5,
                Result.Defeat: 0.0,
            }[game_result]
            efficiency = np.log1p(self.state.score.killed_value_units) - np.log1p(
                self.state.score.lost_minerals_economy
            )
            result = outcome, efficiency

            logger.info(f"{result=}")
            report = self.optimizer.tell(result)
            logger.info(f"{report=}")

    def _macro(self, strategy: StrategyDecision) -> MacroPlan:
        macro_plan = MacroPlan()
        if self.supply_left == 0:
            macro_plan.add(AutoSupply(self.start_location))
        if strategy.morph_drone:
            macro_plan.add(BuildWorkers(to_count=int(self.supply_workers) + 1))
        else:
            macro_plan.add(SpawnController(army_composition_dict=strategy.army_composition))
        if self.can_afford(UnitTypeId.HATCHERY) and self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED):
            macro_plan.add(ExpansionController(to_count=len(self.expansion_locations_list), can_afford_check=False))
        macro_plan.add(GasBuildingController(to_count=1))
        macro_plan.add(UpgradeController(strategy.upgrades, self.start_location))
        return macro_plan
