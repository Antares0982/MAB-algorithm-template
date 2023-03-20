import os
import sys
import unittest


class Montecarlo_test(unittest.TestCase):
    def test_draw(self):
        from MAB_algorithm.arm import armList, heavyTailArm
        from MAB_algorithm.MAB import DSEE, CatoniRobustUCB
        from MAB_algorithm.MAB_MC import MAB_MonteCarlo
        from MAB_algorithm.mabplot import plotResult

        arms = [heavyTailArm(x, 2, float(x/3), 2) for x in range(4)]

        tests = [
            MAB_MonteCarlo(DSEE, arms, w=1),
            MAB_MonteCarlo(
                CatoniRobustUCB, arms,
                v=armList.getmaxVariance(arms)
            )
        ]

        # data = [None]*len(tests)
        # for i, monte in enumerate(tests):
        #     data[i] = monte.to_average_dataframe(
        #         monte.run_monte_carlo_to_list(100, 1000, useCores=8)
        #     )

        # plotResult(data, [t.algorithm.__name__ for t in tests])

        plotResult([x for x in tests], 1000, [
                   x.algorithm.__name__ for x in tests])


if __name__ == "__main__":
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    unittest.main()
