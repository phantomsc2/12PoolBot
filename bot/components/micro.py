import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from itertools import cycle
from typing import Annotated

import numpy as np
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import AMove, UseAbility
from ares.consts import EngagementResult
from cython_extensions import cy_distance_to
from cython_extensions.dijkstra import DijkstraPathing, cy_dijkstra
from leitwerk import Parameter
from loguru import logger
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from scipy.spatial.distance import pdist, squareform

from bot.combat_predictor import CombatPredictor
from bot.components.component import Component
from consts import MAX_MICRO_ACTIONS

Point = tuple[int, int]
HALF = Point2((0.5, 0.5))


class CombatStance(Enum):
    Attack = auto()
    Runby = auto()
    Retreat = auto()


@dataclass(frozen=True)
class MicroParams:
    attack_threshold: Annotated[float, Parameter()]
    supply_confidence_boost: Annotated[float, Parameter(loc=5.0, scale=1.0, min=0.0)]


class Micro(Component):
    def __init__(self) -> None:
        super().__init__()
        self._commit = False
        self.commit_at_supply = 100
        self.commit_cancel_at_supply = 50

    def micro(self, combat: CombatPredictor, params: MicroParams) -> None:

        if not self._commit and self.supply_used > self.commit_at_supply:
            logger.info(f"Reached {self.supply_used} supply, committing hard")
            self._commit = True
        if self._commit and self.supply_used < self.commit_cancel_at_supply:
            logger.info(f"Fell down to {self.supply_used} supply, cancelling commit")
            self._commit = False

        runby = self._runby_pathing()

        self._micro_army(combat, runby, params)
        self._micro_queens()

    def _micro_army(self, combat: CombatPredictor, runby_pathing: DijkstraPathing, params: MicroParams) -> None:
        units = sorted(self.units({UnitTypeId.ZERGLING, UnitTypeId.ROACH, UnitTypeId.MUTALISK}), key=lambda u: u.tag)
        target_units = sorted(combat.enemy_units.not_flying, key=lambda u: u.tag)
        civilians = self.workers

        action_interval = max(1, math.ceil(len(units) / MAX_MICRO_ACTIONS))
        if action_interval > 1:
            logger.info(f"{action_interval=}")

        if not target_units or not civilians:
            for unit in units:
                if unit.is_idle:
                    maneuver = CombatManeuver()
                    maneuver.add(AMove(unit, self.random_scout_target()))
                    self.register_behavior(maneuver)
            return

        attack_targets = [u.position for u in target_units]
        attack_center = _medoid(attack_targets)
        attack_targets.sort(key=lambda t: t.distance_to(attack_center), reverse=True)

        retreat_targets = [w.position for w in civilians]
        retreat_center = _medoid(retreat_targets)
        retreat_targets.sort(key=lambda t: t.distance_to(retreat_center))

        retreat_pathing = cy_dijkstra(self.mediator.get_ground_grid, np.atleast_2d(retreat_targets))

        for unit, target, retreat_target in zip(units, cycle(attack_targets), cycle(retreat_targets)):
            if (self.actual_iteration % action_interval) != (unit.tag % action_interval):
                continue

            maneuver = CombatManeuver()
            outcome = combat.prediction.outcome_for[unit.tag]
            confidence_boost = (self.supply_used / 200.0) * params.supply_confidence_boost

            if self._commit or outcome + params.attack_threshold + confidence_boost > EngagementResult.TIE:
                stance = CombatStance.Attack
            elif not self.mediator.is_position_safe(
                grid=self.mediator.get_ground_grid,
                position=unit.position,
                weight_safety_limit=1.0,
            ):
                stance = CombatStance.Retreat
            else:
                stance = CombatStance.Runby

            if stance == CombatStance.Attack:
                maneuver.add(AMove(unit, target))
            elif stance == CombatStance.Retreat:
                retreat_path_limit = 3
                retreat_path = retreat_pathing.get_path(unit.position, retreat_path_limit)
                retreat_target = Point2(retreat_path[-1]).offset(HALF)
                maneuver.add(UseAbility(AbilityId.MOVE_MOVE, unit, retreat_target))
            else:
                runby_path = runby_pathing.get_path(unit.position, 3)
                runby_point = Point2(runby_path[-1])
                if cy_distance_to(unit.position, runby_point) < 1:
                    maneuver.add(AMove(unit, runby_point))
                else:
                    maneuver.add(UseAbility(AbilityId.MOVE_MOVE, unit, runby_point))
            self.register_behavior(maneuver)

    def _micro_queens(self) -> None:
        queens = sorted(self.mediator.get_own_army_dict[UnitTypeId.QUEEN], key=lambda u: u.tag)
        hatcheries = sorted(self.townhalls, key=lambda u: u.distance_to(self.start_location))
        for queen, hatchery in zip(queens, hatcheries, strict=False):
            maneuver = CombatManeuver()
            queen_position = hatchery.position.towards(self.game_info.map_center, queen.radius + hatchery.radius)
            if queen.energy >= 25 and hatchery.is_ready:
                maneuver.add(UseAbility(AbilityId.EFFECT_INJECTLARVA, queen, hatchery))
            elif queen.distance_to(queen_position) > 1:
                maneuver.add(UseAbility(AbilityId.ATTACK, queen, hatchery))
            self.register_behavior(maneuver)

    def random_scout_target(self, num_attempts=10) -> Point2:
        def sample() -> Point2:
            a = self.game_info.playable_area
            return Point2(np.random.uniform((a.x, a.y), (a.right, a.top)))

        if self.enemy_structures.exists:
            return self.enemy_structures.random.position
        for p in self.enemy_start_locations:
            if not self.is_visible(p):
                return p
        for _ in range(num_attempts):
            target = sample()
            if self.in_pathing_grid(target) and not self.is_visible(target):
                return target
        return sample()

    def _runby_pathing(self) -> DijkstraPathing:
        targets: set[tuple[int, int]] = set()
        for structure in self.enemy_structures:
            targets.update(_structure_perimeter(structure))
        for worker in self.enemy_workers:
            targets.add(worker.position.rounded)
        if not targets:
            targets.add(self.enemy_start_locations[0].rounded)
        cost = self.mediator.get_ground_grid
        target_array = np.atleast_2d(list(targets))
        return cy_dijkstra(cost, target_array)


def _structure_perimeter(structure: Unit) -> Iterable[tuple[int, int]]:
    if structure.is_flying:
        return
    half_extent = structure.footprint_radius
    if structure.position is None or half_extent is None:
        return
    x0, y0 = np.subtract(structure.position, half_extent).astype(int) - 1
    x1, y1 = np.add(structure.position, half_extent).astype(int)

    for x in range(x0, x1 + 1):
        yield x, y0
        yield x, y1

    for y in range(y0 + 1, y0):
        yield x0, y
        yield x1, y


def _medoid(points: Sequence[Point2]) -> Point2:
    distances = squareform(pdist(points), checks=False)
    medoid_index = distances.sum(axis=1).argmin()
    return points[medoid_index]


def _pairwise_distances(positions: Sequence[Point2]) -> np.ndarray:
    return squareform(pdist(positions), checks=False)
