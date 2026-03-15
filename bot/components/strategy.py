from dataclasses import dataclass

import numpy as np
from cython_extensions import cy_unit_pending
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2

from .component import Component


@dataclass
class StrategyDecision:
    build_unit: UnitTypeId
    vespene_target: int
    tech_building_position: Point2


class Strategy(Component):
    def decide_strategy(self) -> StrategyDecision:
        larva_per_second = sum(
            sum(
                (
                    1 / 11 if h.is_ready else 0,
                    3 / 29 if h.has_buff(BuffId.QUEENSPAWNLARVATIMER) else 0,
                )
            )
            for h in self.townhalls
        )
        minerals_for_lings = 50 * 60 * larva_per_second  # maximum we can possibly spend on lings
        max_spending = minerals_for_lings
        should_drone = (
            self.minerals < 150
            and self.state.score.collection_rate_minerals < 1.2 * max_spending  # aim for a 20% surplus
            and self.state.score.food_used_economy < sum(h.ideal_harvesters for h in self.townhalls)
            and not cy_unit_pending(self, UnitTypeId.DRONE)
        )

        early_game = not self.build_order_runner.build_completed
        mine_gas_for_speed = not self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED)

        mutalisk_switch = self.enemy_structures.flying and not self.enemy_structures.not_flying
        build_unit = (
            UnitTypeId.DRONE if should_drone else (UnitTypeId.MUTALISK if mutalisk_switch else UnitTypeId.ZERGLING)
        )
        mine_gas = early_game or mine_gas_for_speed or mutalisk_switch

        vespene_target = 3 if mine_gas else 0
        tech_building_position = self.start_location.towards(self.game_info.map_center, 8)
        return StrategyDecision(
            build_unit=build_unit,
            vespene_target=vespene_target,
            tech_building_position=tech_building_position,
        )
