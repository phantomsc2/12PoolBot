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
    TechUp,
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
    PARAMS_FILE,
    VERSION_FILE,
)
from overlord_drop import OverlordDrop


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

        escalator = OverlordDrop.find_escalator_point(self)
        self.overlord_drop = OverlordDrop(escalator)

        # await self.client.debug_create_unit(
        #     [
        #         [UnitTypeId.OVERLORDTRANSPORT, 1, self.game_info.map_center, 1],
        #         [UnitTypeId.ZERGLING, 8, self.game_info.map_center, 1],
        #         [UnitTypeId.LAIR, 1, self.mediator.get_own_nat, 1],
        #     ]
        # )

    async def on_step(self, iteration: int) -> None:
        await super().on_step(iteration)

        strategy = self.decide_strategy()
        units = self.all_own_units.exclude_type(EXCLUDE_FROM_COMBAT)
        enemy_units = self.all_enemy_units.exclude_type(EXCLUDE_FROM_COMBAT)
        predictor = CombatPredictor(self, units, enemy_units, self.params.combat_predictor)

        self.register_behavior(Mining(workers_per_gas=3 if strategy.gas_count > 0 else 0))

        self.micro(predictor, self.params.micro)

        if self.build_order_runner.build_completed:
            self._macro(strategy)

        self.overlord_drop.on_step(self, strategy.dropperlord_count)

        if (
            self.structure_type_build_progress(UnitTypeId.LAIR) == 1.0
            and self.units({UnitTypeId.OVERLORDTRANSPORT, UnitTypeId.TRANSPORTOVERLORDCOCOON}).amount
            < strategy.dropperlord_count
            and self.can_afford(UnitTypeId.OVERLORDTRANSPORT)
        ):
            overlords = self.units(UnitTypeId.OVERLORD)
            if overlords.exists:
                overlords.closest_to(self.enemy_start_locations[0]).train(UnitTypeId.OVERLORDTRANSPORT)

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

    def _macro(self, strategy: StrategyDecision) -> None:
        plan = MacroPlan()
        if self.supply_left == 0:
            plan.add(AutoSupply(self.start_location))
        plan.add(GasBuildingController(to_count=strategy.gas_count))
        plan.add(UpgradeController(strategy.upgrade_targets, self.start_location))
        for tech in strategy.tech_targets:
            plan.add(TechUp(desired_tech=tech, base_location=self.start_location))
        if strategy.morph_drone:
            plan.add(BuildWorkers(to_count=int(self.supply_workers) + 1))
        else:
            plan.add(SpawnController(army_composition_dict=strategy.army_composition))
        if self.can_afford(UnitTypeId.HATCHERY) and self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED):
            plan.add(ExpansionController(to_count=len(self.expansion_locations_list), can_afford_check=False))
        self.register_behavior(plan)
