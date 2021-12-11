import logging
import time
from multiprocessing import Pool
from typing import Any, Dict, List, Type, TypeVar, Union

import numpy as np
import pandas as pd

from MAB_algorithm.arm import *

try:
    from typing import TYPE_CHECKING
except ImportError:
    TYPE_CHECKING = False


if TYPE_CHECKING:
    from MAB_algorithm.MAB import MABAlgorithm

__all__ = [
    "MAB_MonteCarlo"
]

_T = TypeVar("_T")


class MAB_MonteCarlo(object):

    """
    The class for Monte Carlo experiment of MAB (multi-armed bandit) algorithm.

    Passing any MAB algorithm class name which inherited :class:`MABAlgorithm` to the constructor
    will define a MAB Monte Carlo experiment. Then use :method:`run_monte_carlo_to_list` to start experiment.

    Args:
        algorithm (:type:`Type[MABAlgorithm]`): A class inherited `MABAlgorithm`.
    """

    def __init__(
        self,
        algorithm: Type['MABAlgorithm'],
        arms: List[Arm],
        **kwargs
    ) -> None:
        self.algorithm = algorithm
        self.arms = arms
        self.kwargs = kwargs
        self._optimal_arm_index = self._get_optimal_arm_index()
        self.logger = logging.getLogger(__name__)

    def _get_optimal_arm_index(self):
        return armList.get_optimal_arm_index(self.arms)

    @staticmethod
    def getcolnames(colnames) -> List[str]:
        ans = []
        for i, coln in enumerate(colnames):
            if coln == 'iteration':
                ans.append(colnames[i])
            elif coln == 'chosen_arm':
                ans.append('optimal_arm_chosen_possibility')
            else:
                ans += ['avg_'+coln, 'var_'+coln]
        return ans

    def monte_carlo_avg_result(self, colnames: List[str], data: List[List[Union[float, int]]]) -> dict:
        if not hasattr(self, 'columnNames'):
            self.columnNames = self.getcolnames(colnames)

        ans = []
        for i in range(len(colnames)):
            colname = colnames[i]

            if colname == "iteration":
                ans.append(data[0][i])
                continue

            if colname == "chosen_arm":
                ans.append(np.count_nonzero(
                    [x[i] == self._optimal_arm_index for x in data])/len(data))
                continue

            dt = np.array([x[i] for x in data])
            ans += [np.mean(dt), np.var(dt)]

            if self.logger:
                a = np.count_nonzero(
                    [np.abs(x-ans[-2]) > 3*ans[-1] for x in dt])
                if len(dt) > 20 and a/len(dt) > 0.1:
                    self.logger.warning(
                        f"""Algorithm {self.algorithm}, with argument {str(self.kwargs)}:
                            {colname} of too many samples are far from the average.
                            There are {a} out of {len(dt)} samples are far from average.
                            Please check if the distributions of arms are heavy tailed,
                            or the algorithm is not ideal.""")
        return ans

    def multi_process_monte_carlo_result(
            self,
            colnames: List[str],
            dictdata: List[List[Dict[str, Union[float, int]]]],
            iteration: int,
            needDetails: bool
    ) -> dict:

        ans = []

        row = [dictdata[x][iteration] for x in range(len(dictdata))]

        for i in range(len(colnames)):
            colname = colnames[i]
            if colname == 'iteration':
                ans.append(iteration)
                continue

            if colname == 'chosen_arm':
                arm_chosen = [
                    row[x][i]
                    for x in range(len(row))
                ]
                ans.append(
                    np.count_nonzero([
                        x == self._optimal_arm_index for x in arm_chosen
                    ])/len(arm_chosen)
                )
                continue

            dt = np.array([
                row[x][i]
                for x in range(len(row))
            ])

            ans += [np.mean(dt), np.var(dt)]
            if self.logger:
                a = np.count_nonzero(
                    [np.abs(x-ans[-2]) > 3*ans[-1] for x in dt])
                if len(dt) > 20 and a/len(dt) > 0.1:
                    self.logger.warning(
                        f"""Algorithm {self.algorithm}, with argument {str(self.kwargs)}:
                            {colname} of too many samples are far from the average.
                            There are {a} out of {len(dt)} samples are far from average.
                            Please check if the distributions of arms are heavy tailed,
                            or the algorithm is not ideal.""")

        if needDetails:
            ans['details'] = row
        return ans

    def monte_carlo_gen(
        self,
        repeatTimes: int,
        iterations: int,
        needDetails: bool = False
    ):
        """
        Run a Monte Carlo test. This method returns a generator.
        For multi-processing, use :method:`run_monte_carlo_to_list`.
        """
        if repeatTimes < 1:
            raise ValueError("Repeat times should be at least 1")
        if iterations < 1:
            raise ValueError("Number of iterations must be positive")
        return self._run_single_process(repeatTimes, iterations, needDetails)

    def run_monte_carlo_to_list(
        self,
        repeatTimes: int,
        iterations: int,
        useCores: int = 1
    ) -> List[List[List[Union[float, int]]]]:
        if repeatTimes < 1:
            raise ValueError("Repeat times should be at least 1")
        if iterations < 1:
            raise ValueError("Number of iterations must be positive")
        if useCores < 1:
            raise ValueError("Number of cores to be used should be at least 1")

        return self._run_multi_process(repeatTimes, iterations, useCores)

    def _run_single_process(self, repeatTimes: int, iterations: int, needDetails: bool):
        """
        This method returns a generator yielding answer at each step.
        """
        algs = [
            self.algorithm(
                self.arms, loggerOn=False, **self.kwargs)
            for _ in range(repeatTimes)
        ]

        if not hasattr(self, 'columnNames'):
            self.columnNames = self.getcolnames(algs[0].columnNames)

        self.monte_carlo_iters = [
            alg.run_simulation(iterations)
            for alg in algs
        ]
        colnames = algs[0].columnNames

        for it in range(iterations):
            starttime = time.time()
            data = [x.__next__() for x in self.monte_carlo_iters]
            ans = self.monte_carlo_avg_result(colnames, data)
            if needDetails:
                ans["details"] = data
            endtime = time.time()
            passedTime = int(100*(endtime-starttime))/100
            if self.logger and passedTime > repeatTimes:
                self.logger.warning(f"Single iteration took too long time;\n\
                                    Took {passedTime} seconds at iteration: {it}")

            yield ans

    @staticmethod
    def _list_flatten(lst: List[List[_T]]) -> List[_T]:
        v = lst[0]
        for vv in lst[1:]:
            v = np.append(v, vv, axis=0)
        return v

    def _run_multi_process(
            self,
            repeatTimes: int,
            iterations: int,
            processes: int
    ) -> List[List[List[Union[float, int]]]]:
        sample_per_process = repeatTimes // processes
        residue = repeatTimes - sample_per_process*processes
        sample_numbers = [sample_per_process+1]*residue + \
            [sample_per_process]*(processes-residue)
        if not hasattr(self, 'columnNames'):
            self.columnNames = self.getcolnames(self.algorithm(
                self.arms, loggerOn=False, **self.kwargs).columnNames)

        tuplelist = [(
            self.algorithm,
            self.arms,
            self.kwargs,
            iterations,
            sample_numbers[i]
        ) for i in range(processes)]

        with Pool(processes=processes) as mp:
            dum = np.array(mp.starmap(
                self._run_to_list_with_multi_process, tuplelist))
            # print(dum)
        table: List[List[List[Union[float, int]]]] = self._list_flatten(dum)
        # print(table)

        return table
        # every element in table is an array of algorithm result, need to

    @staticmethod
    def _run_to_list_with_multi_process(
            algorithm: Type['MABAlgorithm'],
            arms: List[Arm],
            kwargs: Dict[str, Any],
            iterations: int,
            size
    ) -> List[List[List[Union[float, int]]]]:
        """Static method to run a single process MAB monte carlo test."""
        return np.array([algorithm(
            arms,
            loggerOn=False,
            **kwargs
        ).run_simulation_tolist(iterations)
            for _ in range(size)])

    def to_average(self, result: List[List[List[Union[float, int]]]]) -> List[List[Union[float, int]]]:
        m = len(result[0])
        n = len(self.columnNames)
        ans = np.zeros((m, n))
        i = 0
        j = 0
        for colname in self.columnNames:
            if colname == "iteration":
                ans[:, i] = np.array(range(m))+1
                i += 1
                j += 1
            elif colname == "optimal_arm_chosen_possibility":
                index = armList.get_optimal_arm_index(self.arms)
                ans[:, i] = 1 - \
                    np.count_nonzero(result[:, :, j]-index, axis=0)/len(result)
                i += 1
                j += 1
            else:
                if colname.startswith("avg"):
                    ans[:, i] = np.mean(result[:, :, j], axis=0)
                else:
                    ans[:, i] = np.var(result[:, :, j], axis=0)
                    j += 1
                i += 1
        return ans

    def to_average_dataframe(self, result: List[List[List[Union[float, int]]]]) -> pd.DataFrame:
        return pd.DataFrame(self.to_average(result), columns=self.columnNames)
