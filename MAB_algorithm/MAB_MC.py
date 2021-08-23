from typing import TYPE_CHECKING, Dict, Generator, Iterator, List, Type, Union

import numpy as np

if TYPE_CHECKING:
    from .arm import *
    from .MAB import *

__all__ = [
    "MAB_MonteCarlo"
]


class MAB_MonteCarlo(object):
    """
    The class for Monte Carlo experiment of MAB (multi-armed bandit) algorithm.

    Passing any MAB algorithm class name which inherited :class:`MABAlgorithm` to the constructor
    will define a MAB Monte Carlo experiment. Then use :method:`run_monte_carlo` to start experiment.

    Args:
        algorithm (:type:`Type[MABAlgorithm]`): A class inherited `MABAlgorithm`.
    """

    def __init__(self, algorithm: Type['MABAlgorithm'], arms: List['Arm'], **kwargs) -> None:
        self.algorithm = algorithm
        self.arms = arms
        self.kwargs = kwargs
        self._optimal_arm_index = self._get_optimal_arm_index()

    def _get_optimal_arm_index(self):
        ans = 0
        optimal_reward = self.arms[0].optimal_rewards()
        for i in range(1, len(self.arms)):
            this_reward = self.arms[i].optimal_rewards()
            if this_reward > optimal_reward:
                optimal_reward = this_reward
                ans = i
        return ans

    def monte_carlo_avg_result(self, data: List[Dict[str, Union[float, int]]]) -> dict:
        ans = {}
        for key in data[0].keys():
            if key == "iteration":
                ans[key] = data[0][key]
                continue
            if key == "chosen_arm":
                ans["optimal_arm_chosen_possibility"] = np.count_nonzero(
                    [x[key] == self._optimal_arm_index for x in data])/len(data)
                continue
            
            ans["avg_"+key] = np.mean([x[key] for x in data])
        return ans

    def run_monte_carlo(self, repeatTimes: int, iterations: int, needDetails: bool = False):
        if repeatTimes < 1:
            raise ValueError("Repeat times should be at least 1")
        if iterations < 1:
            raise ValueError("Number of iterations must be positive")
        if self.kwargs:
            self.monte_carlo_iters = [
                self.algorithm(
                    self.arms, **self.kwargs).run_simulation(iterations)
                for _ in range(repeatTimes)
            ]
        else:
            self.monte_carlo_iters = [
                self.algorithm(self.arms).run_simulation(iterations)
                for _ in range(repeatTimes)
            ]

        for _ in range(iterations):
            data = [x.__next__() for x in self.monte_carlo_iters]
            ans = self.monte_carlo_avg_result(data)
            if needDetails:
                ans["details"] = data
            yield ans
