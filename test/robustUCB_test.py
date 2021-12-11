import os
import sys
import unittest


class robustUCB_test(unittest.TestCase):
    def test_draw(self):
        import time
        from typing import List, Type

        import numpy as np
        from MAB_algorithm import (CatoniRobustUCB, MABAlgorithm, armList,
                                   heavyTailArm, medianRobustUCB, plotResult,
                                   singlePlot, truncatedRobustUCB)

        _MAX_MOMENTORDER = 2
        _ARM_SIZE = 4
        _ITER_TIME = 10000

        arms = [heavyTailArm(i, _MAX_MOMENTORDER, float(i/3), 2)
                for i in range(_ARM_SIZE)]

        v = armList.getmaxVariance(arms)
        print("v=", v)

        algos: List[Type[MABAlgorithm]] = [
            truncatedRobustUCB,
            medianRobustUCB,
            CatoniRobustUCB
        ]
        algoname = [x.__name__ for x in algos]
        kwargss = [
            # assume moment upper bound is same as variance
            {"ve": (_MAX_MOMENTORDER-1)/2, "u": v},
            # assume moment upper bound is same as variance
            {"ve": (_MAX_MOMENTORDER-1)/2, "v": v},
            {"v": v}
        ]

        _SIZE = len(algos)
        ans = [None]*_SIZE

        for i, alg in enumerate(algos):
            r_al = alg(arms, **kwargss[i])

            t0 = time.time()
            ans[i] = r_al.run_to_pdframe(_ITER_TIME)
            t1 = time.time()
            print(f"{alg.__name__} used {t1-t0} seconds")

            singlePlot(ans[i], algoname[i],
                       r_al.gen_regret_ub_curve(_ITER_TIME))

            chosen = [len(arr) for arr in r_al.reward_history]
            self.assertEqual(_ARM_SIZE-1, np.argmax(chosen))

        plotResult(ans, algoname)


if __name__ == "__main__":
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    unittest.main()
