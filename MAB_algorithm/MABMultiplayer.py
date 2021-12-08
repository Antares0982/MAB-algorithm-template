# TODO(Antares): will finish this later.

__all__ = []  # TODO(Antares): delete this

from typing import Callable, Dict, List, Optional, Set, overload

import numpy as np

from MAB_algorithm.arm import *
from MAB_algorithm.MAB import *


class MABMultiplayer(MABAlgorithm):
    def __init__(self, arms: List[Arm], numberOfPlayers: int) -> None:
        if numberOfPlayers <= 1:
            raise ValueError("There should be at least 2 players")
        self._arms = arms
        self.__number_of_players = numberOfPlayers
        self._reward_sum = list(
            map(np.zeros, [len(self._arms)]*numberOfPlayers))
        self._counts = list(map(np.zeros, [len(self._arms)]*numberOfPlayers))

    @property
    def number_of_players(self) -> int:
        return self.__number_of_players

    @number_of_players.setter
    def number_of_players(self, value: int):
        if value < 2:
            raise ValueError("There should be at least 2 players")
        if value > self.__number_of_players:
            self._reward_sum += list(map(np.zeros,
                                     [len(self._arms)]*(value-self.__number_of_players)))
            self._counts += list(map(np.zeros,
                                 [len(self._arms)]*(value-self.__number_of_players)))
        elif value < self.__number_of_players:
            self._reward_sum = self._reward_sum[:value]
            self._counts = self._counts[:value]
        self.__number_of_players = value

    @overload
    def select_arm(self,
                   player: int,
                   mean_estimator: Optional[Callable[..., int]] = None,
                   *args, **kwargs
                   ) -> int:
        ...

    def select_arm(self, *args, **kwargs):
        """
        The method that returns the index of the Arm that players select on the current play.
        This method should be implemented in subclasses.
        Suggested arguments are listed below, and are passed to `select_arm()` by default.

        Args:
            player (:obj:`int`): The index of player.
            count (:obj:`int`): current number of draws.
            mean_estimator (:obj:`function`, optional) gives the mean estimator to select arms.
                Generally, this function is not needed, since the algorithm should
                implement the property :property:`mean`. One should define this function
                if algorithm needs two mean estimator, or the arithmetic mean needs to be
                retained.
            other (`Any`): Any keyword argument needed. If needed, one should also
                override :method:`run_simulation()`.

        Returns:
            :obj:`int`: index of chosen arm.
        """
        raise NotImplementedError

    def run_simulation(self, iterations: int) -> List[dict]:
        if iterations <= 0:
            raise ValueError("Iterations must be positive")

        number_of_arms = len(self._arms)
        results = [[] for i in range(self.number_of_players)]

        optimal_strategy_rewards = [0.0 for i in range(self.number_of_players)]
        collected_rewards = [0.0 for i in range(self.number_of_players)]

        arm_rewards = [self._arms[arm_index].optimal_rewards()
                       for arm_index in range(number_of_arms)]
        arm_rewards.sort(reverse=True)
        optimal_arm_rewards = sum(
            arm_rewards[:self.number_of_players]) / self.number_of_players

        for it in range(iterations):
            arm_selection_player_map: Dict[int, Set[int]] = {}
            reward = []
            count = []
            for player in range(self.number_of_players):
                chosen_arm_index = self.select_arm(
                    player=player,
                    count=it,
                    mean_estimator=None
                )
                if chosen_arm_index not in arm_selection_player_map:
                    arm_selection_player_map[chosen_arm_index] = set()
                arm_selection_player_map[chosen_arm_index].add(player)
        pass  # TODO
