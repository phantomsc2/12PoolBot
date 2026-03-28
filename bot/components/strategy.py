from dataclasses import dataclass
from typing import TypeAlias

from cython_extensions import cy_unit_pending
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId

from bot.components.component import Component

UnitComposition: TypeAlias = dict[UnitTypeId, dict[str, float | int]]


@dataclass(frozen=True)
class StrategyDecision:
    morph_drone: bool
    army_composition: UnitComposition
    gas_count: int
    dropperlord_count: int
    tech_targets: list[UnitTypeId]
    upgrade_targets: list[UpgradeId]


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
        go_upgrades = self.townhalls.amount >= 3 and self.workers.amount >= 32
        make_banes = False
        dropperlord_count = 1 if self.townhalls.amount >= 2 and self.workers.amount >= 16 else 0

        composition: UnitComposition = {}
        if mutalisk_switch:
            composition[UnitTypeId.MUTALISK] = {"proportion": 1.0, "priority": 1}
        elif (
            not self.larva.exists
            and not cy_unit_pending(self, UnitTypeId.QUEEN)
            and self.mediator.get_own_unit_count(unit_type_id=UnitTypeId.QUEEN) < self.townhalls.ready.amount
        ):
            composition[UnitTypeId.QUEEN] = {"proportion": 1.0, "priority": 1}
        elif make_banes:
            composition[UnitTypeId.BANELING] = {"proportion": 0.5, "priority": 1}
            composition[UnitTypeId.ZERGLING] = {"proportion": 0.5, "priority": 1}
        else:
            composition[UnitTypeId.ZERGLING] = {"proportion": 1.0, "priority": 1}

        if not self.build_order_runner.build_completed or not self.already_pending_upgrade(
            UpgradeId.ZERGLINGMOVEMENTSPEED
        ):
            gas_count = 1
        elif (
            mutalisk_switch or (go_upgrades and self.structures(UnitTypeId.EVOLUTIONCHAMBER).idle.exists) or make_banes
        ):
            gas_count = self.workers.amount // 11
        else:
            gas_count = 0
        if (
            self.units({UnitTypeId.OVERLORDTRANSPORT, UnitTypeId.TRANSPORTOVERLORDCOCOON}).amount < dropperlord_count
            and self.vespene < 25
            and self.structure_type_build_progress(UnitTypeId.LAIR) == 1.0
        ):
            gas_count = max(1, gas_count)
        if dropperlord_count > 0 and self.structure_type_build_progress(UnitTypeId.LAIR) == 0.0 and self.vespene < 100:
            gas_count = max(1, gas_count)

        upgrade_targets: set[UpgradeId] = set()
        tech_targets: set[UnitTypeId] = set()
        upgrade_targets.add(UpgradeId.ZERGLINGMOVEMENTSPEED)

        if go_upgrades:
            upgrade_targets.add(UpgradeId.ZERGMELEEWEAPONSLEVEL1)
            if UpgradeId.ZERGMELEEWEAPONSLEVEL1 in self.state.upgrades:
                tech_targets.add(UnitTypeId.LAIR)
                upgrade_targets.add(UpgradeId.ZERGMELEEWEAPONSLEVEL2)
            if UpgradeId.ZERGMELEEWEAPONSLEVEL2 in self.state.upgrades:
                tech_targets.add(UnitTypeId.INFESTATIONPIT)
                tech_targets.add(UnitTypeId.HIVE)
                upgrade_targets.add(UpgradeId.ZERGMELEEWEAPONSLEVEL3)
        if mutalisk_switch:
            tech_targets.add(UnitTypeId.MUTALISK)
            # tech_targets.add(UnitTypeId.SPIRE)
        if make_banes:
            tech_targets.add(UnitTypeId.BANELINGNEST)
        if dropperlord_count > 0:
            tech_targets.add(UnitTypeId.LAIR)

        return StrategyDecision(
            morph_drone=should_drone,
            army_composition=composition,
            gas_count=gas_count,
            dropperlord_count=dropperlord_count,
            tech_targets=list(tech_targets),
            upgrade_targets=list(upgrade_targets),
        )
