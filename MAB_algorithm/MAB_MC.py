import logging
import threading
import time
from multiprocessing import Manager, Pool
from typing import Any, Dict, List, Type, Union

import numpy as np

from MAB_algorithm.arm import *

try:
    from typing import TYPE_CHECKING
except ImportError:
    TYPE_CHECKING = False


if TYPE_CHECKING:
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
    def getcolnames(colnames):
        ans = []
        for i in range(len(colnames)):
            if colnames[i] == 'iteration':
                ans.append(colnames[i])
            elif colnames[i] == 'chosen_arm':
                ans.append('optimal_arm_chosen_possibility')
            else:
                ans += ['avg_'+colnames[i], 'var_'+colnames[i]]
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
        if not hasattr(self, 'columnNames'):
            self.columnNames = self.getcolnames(colnames)

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

    def run_monte_carlo(
        self,
        repeatTimes: int,
        iterations: int,
        needDetails: bool = False,
        useCores: int = 1
    ):
        if repeatTimes < 1:
            raise ValueError("Repeat times should be at least 1")
        if iterations < 1:
            raise ValueError("Number of iterations must be positive")
        if useCores < 1:
            raise ValueError("Number of cores to be used should be at least 1")
        if useCores == 1:
            return self._run_single_process(repeatTimes, iterations, needDetails)
        else:
            return self._run_multi_process(repeatTimes, iterations, needDetails, useCores)

    def _run_single_process(self, repeatTimes: int, iterations: int, needDetails: bool):
        algs = [
            self.algorithm(
                self.arms, loggerOn=False, **self.kwargs)
            for _ in range(repeatTimes)
        ]
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

    def _run_multi_process(
            self,
            repeatTimes: int,
            iterations: int,
            needDetails: bool,
            processes: int
    ):
        sample_per_process = repeatTimes // processes
        residue = repeatTimes - sample_per_process*processes
        sample_numbers = [sample_per_process+1]*residue + \
            [sample_per_process]*(processes-residue)

        slice_index = [0]*(processes+1)
        for i in range(1, processes+1):
            slice_index[i] = slice_index[i-1] + sample_numbers[i-1]

        # process manager
        manager = Manager()
        waitingjobcounts = manager.list(
            [processes for _ in range(iterations)]
        )

        lock = manager.Lock()

        dictdata: List[List[dict]] = [
            [manager.list([]) for _ in range(iterations)]
            for _ in range(repeatTimes)
        ]
        colnames = self.algorithm(
            self.arms,
            loggerOn=False,
            **self.kwargs
        ).columnNames

        tuplelist = [(
            self.algorithm,
            self.arms,
            self.kwargs,
            iterations,
            dictdata[slice_index[i]:slice_index[i+1]],
            waitingjobcounts,
            lock
        ) for i in range(processes)]

        mp = Pool(processes=processes)

        # main thread reads the output
        th = threading.Thread(target=mp.starmap, args=(
            self._run_to_list_with_multi_process, tuplelist))
        th.start()

        for iterindex in range(iterations):
            while waitingjobcounts[iterindex] != 0:
                time.sleep(1)

            yield self.multi_process_monte_carlo_result(
                colnames,
                dictdata,
                iterindex,
                needDetails
            )

        mp.close()

    @staticmethod
    def _run_to_list_with_multi_process(
            algorithm: type['MABAlgorithm'],
            arms: List[Arm],
            kwargs: Dict[str, Any],
            iterations: int,
            dictdata: List[List[dict]],
            waitingjobcounts: List[int],
            lock: threading.Lock
    ):
        """Static method to run a single process MAB monte carlo test."""
        gens = [algorithm(
            arms,
            loggerOn=False,
            **kwargs
        ).run_simulation(iterations)
            for _ in range(len(dictdata))]

        for nowiterindex in range(iterations):
            for i in range(len(gens)):
                dictdata[i][nowiterindex] += gens[i].__next__()

            lock.acquire()
            waitingjobcounts[nowiterindex] -= 1
            lock.release()
