from ares import AresBot, UnitRole
from ares.behaviors.combat.individual import KeepUnitSafe, MoveToSafeTarget, PathUnitToTarget
from cython_extensions import cy_closest_to, cy_distance_to, cy_flood_fill_grid, cy_sorted_by_distance_to
from loguru import logger
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit


class OverlordDropState:
    def __init__(
        self, tag: int, target: Point2, escalator: Point2, passenger_type: UnitTypeId = UnitTypeId.ZERGLING
    ) -> None:
        self._overlord_tag = tag
        self._target = target
        self._escalator = escalator
        self._passenger_tags: set[int] = set()
        self._passenger_type = passenger_type
        self._passenger_role = UnitRole.ATTACKING_TRANSPORT_SQUAD
        self._load_distance = 12.0  # stay this close to escalator point during loading
        self._unload_distance = 3.0  # stay this close to escalator point during unloading
        self._fully_loaded = False

    def on_step(self, bot: AresBot) -> bool:
        dropperlord = bot.unit_tag_dict.get(self._overlord_tag)
        if dropperlord is None:
            for tag in list(self._passenger_tags):
                self._unassign_passenger(bot, tag)
            return False

        if not bot.mediator.is_position_safe(
            grid=bot.mediator.get_air_grid,
            position=dropperlord.position,
            weight_safety_limit=1.0,
        ):
            if dropperlord.health_percentage < 0.2 and bot.in_pathing_grid(dropperlord):
                # emergency evacuation
                dropperlord(AbilityId.UNLOADALLAT_OVERLORD, dropperlord.position)
            else:
                bot.register_behavior(KeepUnitSafe(unit=dropperlord, grid=bot.mediator.get_air_grid))
            return True

        passengers_on_map: list[Unit] = []
        for tag in list(self._passenger_tags):
            unit = bot.unit_tag_dict.get(tag)
            if unit is not None:
                if self._is_candidate(bot, unit):
                    passengers_on_map.append(unit)
                else:
                    self._unassign_passenger(bot, tag)
            elif tag not in dropperlord.passengers_tags:
                self._unassign_passenger(bot, tag)

        target_height = bot.get_terrain_height(self._target)

        if not self._fully_loaded:
            passengers_needed = dropperlord.cargo_max - len(self._passenger_tags)
            if passengers_needed > 0:
                self._assign_passengers(bot, passengers_needed, dropperlord.position)

            if bot.get_terrain_height(dropperlord) == target_height:
                # move to low ground
                bot.register_behavior(
                    MoveToSafeTarget(unit=dropperlord, grid=bot.mediator.get_air_grid, target=bot.game_info.map_center)
                )
            elif len(passengers_on_map) > 0:
                # load passengers
                closest_passenger = cy_closest_to(dropperlord.position, passengers_on_map)
                if cy_distance_to(dropperlord.position, self._escalator) < self._load_distance:
                    dropperlord.smart(closest_passenger)
                else:
                    bot.register_behavior(
                        MoveToSafeTarget(unit=dropperlord, grid=bot.mediator.get_air_grid, target=self._escalator)
                    )
                for unit in passengers_on_map:
                    if cy_distance_to(unit.position, dropperlord.position) < 1:
                        unit.smart(dropperlord)
                    else:
                        bot.register_behavior(
                            PathUnitToTarget(unit=unit, grid=bot.mediator.get_ground_grid, target=dropperlord.position)
                        )
            elif dropperlord.cargo_left == 0 and dropperlord.cargo_max > 0:
                logger.info(f"{dropperlord=} fully loaded")
                self._fully_loaded = True

        else:
            # unload at target
            for unit in passengers_on_map:
                self._unassign_passenger(bot, unit.tag)
            if dropperlord.cargo_used == 0:
                logger.info(f"{dropperlord=} finished drop")
                return False
            elif bot.get_terrain_height(dropperlord) == target_height and bot.in_pathing_grid(dropperlord):
                dropperlord(AbilityId.UNLOADALLAT_OVERLORD, dropperlord.position)
            elif cy_distance_to(dropperlord.position, self._escalator) < self._unload_distance:
                # force the overlord in towards pathable ground
                bot.register_behavior(
                    PathUnitToTarget(unit=dropperlord, grid=bot.mediator.get_air_grid, target=self._target)
                )
            else:
                bot.register_behavior(
                    MoveToSafeTarget(unit=dropperlord, grid=bot.mediator.get_air_grid, target=self._escalator)
                )

        return True

    def _unassign_passenger(self, bot: AresBot, tag: int) -> None:
        bot.mediator.clear_role(tag=tag)
        self._passenger_tags.discard(tag)

    def _is_candidate(self, bot: AresBot, unit: Unit) -> bool:
        return (
            unit.type_id == self._passenger_type
            and bot.get_terrain_height(unit) < bot.get_terrain_height(self._target)
            and bot.mediator.is_position_safe(
                grid=bot.mediator.get_ground_grid,
                position=unit.position,
                weight_safety_limit=1.0,
            )
        )

    def _assign_passengers(self, bot: AresBot, count: int, near: Point2) -> None:
        candidates = [
            u
            for u in bot.units
            if (self._is_candidate(bot, u) and u.tag not in bot.mediator.get_unit_role_dict[self._passenger_role])
        ]
        candidates = cy_sorted_by_distance_to(candidates, near)
        for passenger in candidates[:count]:
            bot.mediator.assign_role(tag=passenger.tag, role=self._passenger_role)
            self._passenger_tags.add(passenger.tag)


class OverlordDrop:
    def __init__(self, escalator: Point2) -> None:
        self._escalator = escalator
        self._active_drops: dict[int, OverlordDropState] = {}

    def on_step(self, bot: AresBot, active_drops_max: int) -> None:
        if len(self._active_drops) < active_drops_max:
            self._assign_new_drop(bot, bot.enemy_start_locations[0], self._escalator)
        for tag, drop in list(self._active_drops.items()):
            is_active = drop.on_step(bot)
            if not is_active:
                logger.info(f"drop finished for {tag=}")
                del self._active_drops[tag]

    def _assign_new_drop(self, bot: AresBot, target: Point2, escalator: Point2) -> None:
        overlords = bot.units(UnitTypeId.OVERLORDTRANSPORT).tags_not_in(self._active_drops)
        if not overlords.exists:
            return
        overlord = cy_closest_to(target, overlords)
        logger.info(f"queueing drop for {overlord=}")
        self._active_drops[overlord.tag] = OverlordDropState(overlord.tag, target, escalator)

    @classmethod
    def find_escalator_point(cls, bot: AresBot) -> Point2:
        enemy_main = bot.enemy_start_locations[0]
        enemy_main_points = cy_flood_fill_grid(
            start_point=enemy_main,
            terrain_grid=bot.game_info.terrain_height.data_numpy.T,
            pathing_grid=bot.game_info.pathing_grid.data_numpy.T,
            max_distance=40,
            cutoff_points=set(),
        )
        if len(enemy_main_points) == 0:
            return enemy_main

        enemy_ramp = bot.mediator.get_enemy_ramp.top_center

        def score(p: tuple[int, int]) -> float:
            return cy_distance_to(p, enemy_ramp) - cy_distance_to(p, bot.game_info.map_center)

        best = max(enemy_main_points, key=score)
        return Point2(best).towards(enemy_main, 1.0)
