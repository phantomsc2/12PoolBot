import math
from dataclasses import dataclass
from enum import Enum, auto
from itertools import chain, cycle
from typing import Iterable

import numpy as np
from ares.consts import DEBUG, EngagementResult
from cython_extensions.dijkstra import cy_dijkstra  # type: ignore
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from ..action import Action, AttackMove, HoldPosition, Move, UseAbility
from ..combat_predictor import CombatPredictor
from .component import Component

Point = tuple[int, int]
HALF = Point2((0.5, 0.5))


class CombatAction(Enum):
    Attack = auto()
    Hold = auto()
    Retreat = auto()


class Micro(Component):
    _action_cache: dict[int, Action] = {}

    def micro(self, combat: CombatPredictor, pathing: np.ndarray, supply_used: int) -> Iterable[Action]:
        return chain(
            self.micro_army(combat, pathing, supply_used),
            self.micro_queens(),
        )

    def micro_army(self, combat: CombatPredictor, pathing: np.ndarray, supply_used: int) -> Iterable[Action]:
        units = sorted(self.units({UnitTypeId.ZERGLING, UnitTypeId.ROACH, UnitTypeId.MUTALISK}), key=lambda u: u.tag)
        target_units = sorted(combat.enemy_units.not_flying, key=lambda u: u.tag)
        civilians = self.workers

        if not target_units or not civilians:
            for unit in units:
                if unit.is_idle:
                    yield AttackMove(unit, self.random_scout_target())
            return

        attack_targets = [u.position for u in target_units]
        attack_center = Point2(np.median(np.array(attack_targets), axis=0))
        attack_targets.sort(key=lambda t: t.distance_to(attack_center), reverse=True)

        retreat_targets = [w.position for w in civilians]
        retreat_center = Point2(np.median(np.array(retreat_targets), axis=0))
        retreat_targets.sort(key=lambda t: t.distance_to(retreat_center))

        attack_pathing = cy_dijkstra(
            pathing,
            np.array(attack_targets, dtype=np.intp),
        )
        retreat_pathing = cy_dijkstra(
            pathing,
            np.array(retreat_targets, dtype=np.intp),
        )

        if self.config[DEBUG]:
            self.mediator.get_map_data_object.draw_influence_in_game(pathing)

        for unit, target, retreat_target in zip(units, cycle(attack_targets), cycle(retreat_targets)):
            p = unit.position.rounded
            attack_path_limit = 5
            attack_path = attack_pathing.get_path(p, attack_path_limit)
            outcome = combat.prediction.outcome_for[unit.tag]

            bias = 0.0
            bias += 4 * (supply_used / 200)**2
            if outcome + bias > EngagementResult.TIE:
                combat_action = CombatAction.Attack
            elif pathing[p] > 1:
                combat_action = CombatAction.Retreat
            else:
                combat_action = CombatAction.Hold

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
                action = HoldPosition(unit)

            is_repeated = action == self._action_cache.get(unit.tag, None)
            if action and not is_repeated:
                self._action_cache[unit.tag] = action
                yield action

    def micro_queens(self) -> Iterable[Action]:
        queens = sorted(self.mediator.get_own_army_dict[UnitTypeId.QUEEN], key=lambda u: u.tag)
        hatcheries = sorted(self.townhalls, key=lambda u: u.distance_to(self.start_location))
        for queen, hatchery in zip(queens, hatcheries):
            queen_position = hatchery.position.towards(self.game_info.map_center, queen.radius + hatchery.radius)
            if 25 <= queen.energy and hatchery.is_ready:
                yield UseAbility(queen, AbilityId.EFFECT_INJECTLARVA, hatchery)
            elif 1 < queen.distance_to(queen_position):
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
