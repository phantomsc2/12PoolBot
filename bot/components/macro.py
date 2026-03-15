from collections.abc import Iterable
from functools import cached_property

from ares.consts import ALL_STRUCTURES
from sc2.dicts.unit_train_build_abilities import TRAIN_INFO
from sc2.dicts.unit_trained_from import UNIT_TRAINED_FROM
from sc2.dicts.upgrade_researched_from import UPGRADE_RESEARCHED_FROM
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from ..action import Action, Build, DoNothing, Research, Train
from .component import Component


class Macro(Component):
    def macro(self, unit: UnitTypeId) -> Iterable[Action]:
        yield (
            self.wait_for_build_order_completion()
            or self.build_unit(unit, limit=1 if unit == UnitTypeId.DRONE else None)
            or self.build_unit(
                UnitTypeId.QUEEN, limit=self.townhalls.amount - len(self.mediator.get_own_army_dict[UnitTypeId.QUEEN])
            )
            or self.make_tech(unit)
            or self.research_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED)
            or self.build_unit(UnitTypeId.OVERLORD, limit=1 if self.supply_left <= 0 else 0)
            or self.expand()
            or DoNothing()
        )

    @cached_property
    def tech_building_position(self):
        return self.start_location.towards(self.game_info.map_center, 8)

    def wait_for_build_order_completion(self) -> Action | None:
        if self.build_order_runner.build_completed:
            return None
        return DoNothing()

    def expand(self) -> Action | None:
        if not self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED) or not (
            target := self.get_next_free_expansion()
        ):
            return None
        return self.build_unit(UnitTypeId.HATCHERY, target=target, limit=1)

    def make_tech(self, unit: UnitTypeId) -> Action | None:
        build_structures: set[UnitTypeId] = set()
        if unit == UnitTypeId.ZERGLING:
            build_structures.add(UnitTypeId.SPAWNINGPOOL)
        elif unit == UnitTypeId.MUTALISK:
            build_structures.add(UnitTypeId.SPAWNINGPOOL)
            build_structures.add(UnitTypeId.LAIR)
            build_structures.add(UnitTypeId.SPIRE)

        for requirement in build_structures:
            if action := self.build_unit(
                requirement,
                target=self.tech_building_position,
                limit=1 - len(self.mediator.get_own_structures_dict[requirement]),
            ):
                return action
        return None

    def get_next_free_expansion(self) -> Point2 | None:
        taken = {th.position for th in self.townhalls}
        return next((p for p, d in self.mediator.get_own_expansions if p not in taken), None)

    def find_trainer(
        self,
        type_id: UnitTypeId | UpgradeId,
        target: Point2 | None = None,
    ) -> Unit | None:
        def filter_trainer(t: Unit) -> bool:
            # TODO: handle reactors
            return not (t.type_id in ALL_STRUCTURES and not t.is_idle)

        def trainer_priority(t: Unit) -> float:
            return -t.position.distance_to(target or self.start_location)

        trainer_types = (
            UNIT_TRAINED_FROM[type_id] if isinstance(type_id, UnitTypeId) else {UPGRADE_RESEARCHED_FROM[type_id]}
        )

        def trainer_pool(t: UnitTypeId) -> Units:
            if t in ALL_STRUCTURES:
                return self.mediator.get_own_structures_dict[t]
            else:
                return self.mediator.get_own_army_dict[t]

        trainers = (t for t_id in trainer_types for t in trainer_pool(t_id) if filter_trainer(t))

        return max(trainers, key=trainer_priority, default=None)

    def build_unit(self, unit: UnitTypeId, target: Point2 | None = None, limit: int | None = None) -> Action | None:
        if (
            self.supply_left < self.calculate_supply_cost(unit)
            or limit is not None
            and limit <= self.already_pending(unit)
            or not (trainer := self.find_trainer(unit, target=target))
            or not self.can_afford(unit)
            or self.tech_requirement_progress(unit) < 1
        ):
            return None
        elif TRAIN_INFO[trainer.type_id][unit].get("requires_placement_position", False):
            return Build(trainer, unit, target)
        return Train(trainer, unit)

    def research_upgrade(self, upgrade: UpgradeId) -> Action | None:
        if self.already_pending_upgrade(upgrade) or not (researcher := self.find_trainer(upgrade)):
            return None
        elif not self.can_afford(upgrade):
            # return DoNothing()
            return None
        return Research(researcher, upgrade)
