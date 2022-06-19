import os
import sys
import unittest


class normalArm_test(unittest.TestCase):
    def test_draw(self):
        import time

        import numpy as np
        from MAB_algorithm import normArm

        _DRAWTIMES = 1000000

        means = [i*0.05+0.65 for i in range(8)]
        arms = [normArm(i, means[i], 2) for i in range(len(means))]

        for a in arms:
            time0 = time.time()
            lst = a.draw(_DRAWTIMES)
            time1 = time.time()
            me: float = np.mean(lst)
            ve = np.var(lst)

            info = f"Arm with mean: {a.mean}, variance: {a.variance()}, \
                drawn {_DRAWTIMES}, average: {me}, var: {ve}, \
                average draw time: {(time1-time0)/_DRAWTIMES}"
            info = ' '.join(info.split())
            print(info)
            self.assertAlmostEqual(a.mean, me, delta=0.03) # may fail
            self.assertAlmostEqual(a.variance(), ve, delta=0.02) # may fail


if __name__ == "__main__":
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    unittest.main()
