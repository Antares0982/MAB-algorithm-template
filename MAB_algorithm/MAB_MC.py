import logging
import time
from typing import TYPE_CHECKING, Dict, List, Type, Union

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
        self.logger = logging.getLogger(__name__)

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
            dt = np.array([x[key] for x in data])
            ans["avg_"+key] = np.mean(dt)
            ans["var_"+key] = np.var(dt)
            if self.logger:
                a = np.count_nonzero(
                    [np.abs(x-ans["avg_"+key]) > 3*ans["var_"+key] for x in dt])
                if a/len(dt) > 0.003:
                    self.logger.warning(
                        f"""Algorithm: {self.algorithm}, with argument:{str(self.kwargs)}
                            Too many samples' {key} are far from the average.
                            There are {a} out of {len(dt)} samples are too far.
                            Please check if the distribution of arms are heavy tailed,
                            or the algorithm is not ideal.""")
        return ans

    def run_monte_carlo(self, repeatTimes: int, iterations: int, needDetails: bool = False):
        if repeatTimes < 1:
            raise ValueError("Repeat times should be at least 1")
        if iterations < 1:
            raise ValueError("Number of iterations must be positive")
        if self.kwargs:
            self.monte_carlo_iters = [
                self.algorithm(
                    self.arms, loggerOn=False, **self.kwargs).run_simulation(iterations)
                for _ in range(repeatTimes)
            ]
        else:
            self.monte_carlo_iters = [
                self.algorithm(
                    self.arms, loggerOn=False).run_simulation(iterations)
                for _ in range(repeatTimes)
            ]

        for it in range(iterations):
            starttime = time.time()
            data = [x.__next__() for x in self.monte_carlo_iters]
            ans = self.monte_carlo_avg_result(data)
            if needDetails:
                ans["details"] = data
            endtime = time.time()
            passedTime = int(100*(endtime-starttime))/100
            if self.logger and passedTime > 5:
                self.logger.warning(f"Single iteration took too long time;\n\
                                    Took {passedTime} seconds at iteration: {it}")

            yield ans
