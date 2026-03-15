from abc import ABC, abstractmethod
from dataclasses import dataclass

from ares import AresBot
from ares.consts import UnitRole
from loguru import logger
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit


class Action(ABC):
    @abstractmethod
    async def execute(self, bot: AresBot) -> bool: ...


class DoNothing(Action):
    async def execute(self, bot: AresBot) -> bool:
        return True


@dataclass(frozen=True)
class AttackMove(Action):
    unit: Unit
    target: Point2

    async def execute(self, bot: AresBot) -> bool:
        return self.unit.attack(self.target)


@dataclass(frozen=True)
class Move(Action):
    unit: Unit
    target: Point2

    async def execute(self, bot: AresBot) -> bool:
        return self.unit.move(self.target)


@dataclass(frozen=True)
class HoldPosition(Action):
    unit: Unit

    async def execute(self, bot: AresBot) -> bool:
        return self.unit.stop()


@dataclass(frozen=True)
class UseAbility(Action):
    unit: Unit
    ability: AbilityId
    target: Point2 | None = None

    async def execute(self, bot: AresBot) -> bool:
        return self.unit(self.ability, target=self.target)


@dataclass(frozen=True)
class Build(Action):
    unit: Unit
    type_id: UnitTypeId
    near: Point2

    async def execute(self, bot: AresBot) -> bool:
        logger.info(self)
        bot.mediator.assign_role(tag=self.unit.tag, role=UnitRole.PERSISTENT_BUILDER)
        if placement := await bot.find_placement(self.type_id, near=self.near):
            return self.unit.build(self.type_id, placement)
        else:
            return False


@dataclass(frozen=True)
class Train(Action):
    trainer: Unit
    unit: UnitTypeId

    async def execute(self, bot: AresBot) -> bool:
        return self.trainer.train(self.unit)


@dataclass(frozen=True)
class Research(Action):
    researcher: Unit
    upgrade: UpgradeId

    async def execute(self, bot: AresBot) -> bool:
        return self.researcher.research(self.upgrade)
