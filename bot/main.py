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

from bot.components.micro import Micro, MicroParams
from bot.components.strategy import Strategy, StrategyDecision
from bot.consts import (
    PARAMS_FILE,
    VERSION_FILE,
)
from bot.overlord_drop import OverlordDrop


@dataclass(frozen=True)
class BotParams:
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

        logger.info(f"{self.optimizer.mean=}")
        logger.info(f"{self.optimizer.scale_marginal=}")

        if VERSION_FILE.exists():
            version = VERSION_FILE.read_text()
            logger.info(f"{version=}")

        escalator = OverlordDrop.find_escalator_point(self)
        self.overlord_drop = OverlordDrop(escalator)

        # await self.client.debug_create_unit(
        #     [
        #         [UnitTypeId.ZERGLING, 6, self.game_info.map_center, 1],
        #         [UnitTypeId.ZERGLING, 8, self.game_info.map_center, 2],
        #     ]
        # )

    async def on_step(self, iteration: int) -> None:
        await super().on_step(iteration)

        strategy = self.decide_strategy()

        self.register_behavior(Mining(workers_per_gas=3 if strategy.gas_count > 0 else 0))

        self.micro(self.params.micro)

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
        mutalisk_plan = UnitTypeId.MUTALISK in strategy.army_composition
        has_lair = self.townhalls(UnitTypeId.LAIR).exists
        handled_techs: set[UnitTypeId] = set()

        if self.supply_left == 0:
            plan.add(AutoSupply(self.start_location))
        plan.add(GasBuildingController(to_count=strategy.gas_count))

        if mutalisk_plan:
            if UnitTypeId.LAIR in strategy.tech_targets and self.minerals >= 150:
                plan.add(TechUp(desired_tech=UnitTypeId.LAIR, base_location=self.start_location))
                handled_techs.add(UnitTypeId.LAIR)
            if UnitTypeId.SPIRE in strategy.tech_targets:
                plan.add(TechUp(desired_tech=UnitTypeId.SPIRE, base_location=self.start_location))
                handled_techs.add(UnitTypeId.SPIRE)

        plan.add(UpgradeController(strategy.upgrade_targets, self.start_location))
        for tech in strategy.tech_targets:
            if tech in handled_techs:
                continue
            plan.add(TechUp(desired_tech=tech, base_location=self.start_location))
        if (
            UnitTypeId.LAIR in strategy.tech_targets
            and self.vespene >= 100
            and self.structure_type_build_progress(UnitTypeId.LAIR) == 0.0
        ):
            # prioritize lair
            pass
        else:
            if strategy.morph_drone:
                plan.add(BuildWorkers(to_count=int(self.supply_workers) + 1))
            # Work around an ares-sc2 TechUp bug for zerg larva units by only
            # handing mutalisk production to SpawnController after lair tech exists.
            elif not mutalisk_plan or has_lair:
                plan.add(SpawnController(army_composition_dict=strategy.army_composition))
            if self.can_afford(UnitTypeId.HATCHERY) and self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED):
                plan.add(ExpansionController(to_count=len(self.expansion_locations_list), can_afford_check=False))
        self.register_behavior(plan)
