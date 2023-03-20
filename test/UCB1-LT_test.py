import os
import sys
import unittest


class UCB1lt_test(unittest.TestCase):
    def test_draw(self):
        import time

        from MAB_algorithm import UCB1LT, BernoulliArm
        from MAB_algorithm.MAB_MC import MAB_MonteCarlo
        from MAB_algorithm.mabplot import plotResult

        _REPEATTIMES = 100
        _DRAWTIMES = 100000
        _THREADS = 8

        means = [i*0.1+0.2 for i in range(8)]  # max: 0.9, min: 0.2
        arms = [BernoulliArm(i, means[i]) for i in range(len(means))]

        ucb1ltMonteCarlo = MAB_MonteCarlo(UCB1LT, arms, zeta=1, u0=10)
        timeStart = time.time()
        ucb1pdframe = ucb1ltMonteCarlo.to_average_dataframe(
            ucb1ltMonteCarlo.run_monte_carlo_to_list(_REPEATTIMES, _DRAWTIMES, useCores=_THREADS))
        timeEnd = time.time()
        print(
            f"Time consumed for {_REPEATTIMES} times Monte Carlo of UCB1LT (T={_DRAWTIMES}, using {_THREADS} threads): {timeEnd-timeStart} seconds")

        plotResult([ucb1pdframe], ["UCB1LT Monte Carlo"])


if __name__ == "__main__":
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    unittest.main()
