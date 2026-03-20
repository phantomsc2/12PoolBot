from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from itertools import chain, cycle
from typing import Annotated

import numpy as np
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

from bot.action import Action, AttackMove, Move, UseAbility
from bot.combat_predictor import CombatPredictor
from bot.components.component import Component

Point = tuple[int, int]
HALF = Point2((0.5, 0.5))


class CombatAction(Enum):
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
        self._action_cache: dict[int, Action] = {}
        self._commit = False
        self.commit_at_supply = 100
        self.commit_cancel_at_supply = 50

    def micro(self, combat: CombatPredictor, params: MicroParams) -> Iterable[Action]:

        if not self._commit and self.supply_used > self.commit_at_supply:
            logger.info(f"Reached {self.supply_used} supply, committing hard")
            self._commit = True
        if self._commit and self.supply_used < self.commit_cancel_at_supply:
            logger.info(f"Fell down to {self.supply_used} supply, cancelling commit")
            self._commit = False

        runby = self._runby_pathing()
        return chain(
            self.micro_army(combat, runby, params),
            self.micro_queens(),
        )

    def micro_army(
        self, combat: CombatPredictor, runby_pathing: DijkstraPathing, params: MicroParams
    ) -> Iterable[Action]:
        units = sorted(self.units({UnitTypeId.ZERGLING, UnitTypeId.ROACH, UnitTypeId.MUTALISK}), key=lambda u: u.tag)
        target_units = sorted(combat.enemy_units.not_flying, key=lambda u: u.tag)
        civilians = self.workers

        if not target_units or not civilians:
            for unit in units:
                if unit.is_idle:
                    yield AttackMove(unit, self.random_scout_target())
            return

        attack_targets = [u.position for u in target_units]
        attack_center = _medoid(attack_targets)
        attack_targets.sort(key=lambda t: t.distance_to(attack_center), reverse=True)

        retreat_targets = [w.position for w in civilians]
        retreat_center = _medoid(retreat_targets)
        retreat_targets.sort(key=lambda t: t.distance_to(retreat_center))

        attack_pathing = cy_dijkstra(self.mediator.get_ground_grid, np.atleast_2d(attack_targets))
        retreat_pathing = cy_dijkstra(self.mediator.get_ground_grid, np.atleast_2d(retreat_targets))

        action: Action
        for unit, target, retreat_target in zip(units, cycle(attack_targets), cycle(retreat_targets)):
            p = unit.position.rounded
            attack_path_limit = 3
            attack_path = attack_pathing.get_path(p, attack_path_limit)
            outcome = combat.prediction.outcome_for[unit.tag]
            confidence_boost = (self.supply_used / 200.0) * params.supply_confidence_boost

            if self._commit or outcome + params.attack_threshold + confidence_boost > EngagementResult.TIE:
                combat_action = CombatAction.Attack
            elif not self.mediator.is_position_safe(
                grid=self.mediator.get_ground_grid,
                position=unit.position,
                weight_safety_limit=1.0,
            ):
                combat_action = CombatAction.Retreat
            else:
                combat_action = CombatAction.Runby

            if combat_action == CombatAction.Attack:
                if len(attack_path) < 2:
                    action = AttackMove(unit, Point2(target))
                else:
                    action = AttackMove(unit, Point2(attack_path[-1]).offset(HALF))
            elif combat_action == CombatAction.Retreat:
                retreat_path_limit = 3
                retreat_path = retreat_pathing.get_path(p, retreat_path_limit)
                if len(retreat_path) < 2:
                    action = Move(unit, Point2(retreat_target))
                elif len(retreat_path) < retreat_path_limit:
                    action = AttackMove(unit, Point2(retreat_path[-1]).offset(HALF))
                else:
                    action = Move(unit, Point2(retreat_path[-1]).offset(HALF))
            else:
                runby_path = runby_pathing.get_path(unit.position, 3)
                runby_point = Point2(runby_path[-1])
                if cy_distance_to(unit.position, runby_point) < 1:
                    action = AttackMove(unit, runby_point)
                else:
                    action = Move(unit, runby_point)

            is_repeated = action == self._action_cache.get(unit.tag, None)
            if action and not is_repeated:
                self._action_cache[unit.tag] = action
                yield action

    def micro_queens(self) -> Iterable[Action]:
        queens = sorted(self.mediator.get_own_army_dict[UnitTypeId.QUEEN], key=lambda u: u.tag)
        hatcheries = sorted(self.townhalls, key=lambda u: u.distance_to(self.start_location))
        for queen, hatchery in zip(queens, hatcheries, strict=False):
            queen_position = hatchery.position.towards(self.game_info.map_center, queen.radius + hatchery.radius)
            if queen.energy >= 25 and hatchery.is_ready:
                yield UseAbility(queen, AbilityId.EFFECT_INJECTLARVA, hatchery)
            elif queen.distance_to(queen_position) > 1:
                yield AttackMove(queen, queen_position)

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
