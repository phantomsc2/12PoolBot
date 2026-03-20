from dataclasses import dataclass
from typing import TypeAlias

from cython_extensions import cy_unit_pending
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId

from .component import Component

UnitComposition: TypeAlias = dict[UnitTypeId, dict[str, float | int]]


@dataclass(frozen=True)
class StrategyDecision:
    morph_drone: bool
    army_composition: UnitComposition
    vespene_target: int
    upgrades: list[UpgradeId]


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
        should_drone = (
            self.minerals < 150
            and self.state.score.collection_rate_minerals < 1.2 * minerals_for_lings  # aim for a 20% surplus
            and self.state.score.food_used_economy < sum(h.ideal_harvesters for h in self.townhalls)
            and not cy_unit_pending(self, UnitTypeId.DRONE)
        )

        mutalisk_switch = self.enemy_structures.flying and not self.enemy_structures.not_flying

        composition: UnitComposition = {}
        if mutalisk_switch:
            composition[UnitTypeId.MUTALISK] = {"proportion": 1.0, "priority": 1}
        elif (
            not self.larva.exists
            and not cy_unit_pending(self, UnitTypeId.QUEEN)
            and self.mediator.get_own_unit_count(unit_type_id=UnitTypeId.QUEEN) < self.townhalls.amount
        ):
            composition[UnitTypeId.QUEEN] = {"proportion": 1.0, "priority": 1}
        else:
            composition[UnitTypeId.ZERGLING] = {"proportion": 1.0, "priority": 1}

        early_game = not self.build_order_runner.build_completed
        mine_gas_for_speed = not self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED)
        mine_gas = early_game or mine_gas_for_speed or mutalisk_switch
        vespene_target = 3 if mine_gas else 0

        upgrades = [UpgradeId.ZERGLINGMOVEMENTSPEED]

        return StrategyDecision(
            morph_drone=should_drone,
            army_composition=composition,
            vespene_target=vespene_target,
            upgrades=upgrades,
        )
