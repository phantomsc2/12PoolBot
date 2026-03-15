from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from itertools import chain
from typing import Annotated

import numpy as np
from ares import AresBot
from ares.consts import EngagementResult
from leitwerk import Parameter
from sc2.units import Units
from sklearn.metrics import pairwise_distances


def graph_components(adjacency_matrix: np.ndarray) -> list[set[int]]:
    components = list[set[int]]()
    for i in range(adjacency_matrix.shape[0]):
        connected_to = set(np.nonzero(adjacency_matrix[i, :i])[0])
        new_component = {i}
        for c in components:
            if c & connected_to:
                components.remove(c)
                new_component.update(c)
        components.append(new_component)
    return components


class CombatOutcome(Enum):
    Victory = auto()
    Defeat = auto()
    Draw = auto()


@dataclass(frozen=True)
class CombatPrediction:
    outcome: EngagementResult
    outcome_for: Mapping[int, EngagementResult]


@dataclass(frozen=True)
class CombatPredictorParams:
    contact_range_internal: Annotated[float, Parameter(loc=6, scale=1, min=0)]
    contact_range: Annotated[float, Parameter(loc=12, scale=2, min=0)]


class CombatPredictor:
    def __init__(self, bot: AresBot, units: Units, enemy_units: Units, params: CombatPredictorParams):
        self.bot = bot
        self.units = units
        self.enemy_units = enemy_units
        self.params = params
        self.prediction = self._predict()

    def _predict(self) -> CombatPrediction:
        n = len(self.units)
        len(self.enemy_units)

        units = list(chain(self.units, self.enemy_units))

        if not any(units):
            return CombatPrediction(EngagementResult.TIE, {})
        elif not any(self.units):
            return CombatPrediction(EngagementResult.LOSS_OVERWHELMING, {})
        elif not any(self.enemy_units):
            return CombatPrediction(EngagementResult.VICTORY_OVERWHELMING, {})

        positions = [u.position for u in self.units]
        enemy_positions = [u.position for u in self.enemy_units]
        distance_matrix = pairwise_distances(positions, enemy_positions)

        contact = np.where(distance_matrix < self.params.contact_range, 1, 0)
        contact_own = np.zeros((n, n))
        contact_enemy = np.where(pairwise_distances(enemy_positions) < self.params.contact_range_internal, 1, 0)

        adjacency_matrix = np.block([[contact_own, contact], [contact.T, contact_enemy]])
        components = graph_components(adjacency_matrix)

        simulator_kwargs = dict(
            good_positioning=False,
            workers_do_no_damage=False,
        )
        outcome = self.bot.mediator.can_win_fight(
            own_units=self.units, enemy_units=self.enemy_units, timing_adjust=False, **simulator_kwargs
        )

        outcome_for = dict[int, EngagementResult]()
        for component in components:
            local_units = [units[i] for i in component]
            local_own = list(filter(lambda u: u.is_mine, local_units))
            local_enemies = list(filter(lambda u: u.is_enemy, local_units))
            if not any(local_own):
                local_outcome = EngagementResult.LOSS_OVERWHELMING if any(local_enemies) else EngagementResult.TIE
            elif not any(local_enemies):
                local_outcome = EngagementResult.VICTORY_OVERWHELMING
            else:
                local_outcome = self.bot.mediator.can_win_fight(
                    own_units=local_own,
                    enemy_units=local_enemies,
                    timing_adjust=True,
                    **simulator_kwargs,
                )

            for u in local_own:
                outcome_for[u.tag] = local_outcome

        return CombatPrediction(outcome, outcome_for)
