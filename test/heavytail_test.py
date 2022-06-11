import os
import sys
import unittest


class heavytail_test(unittest.TestCase):
    def test_draw(self):
        import time

        import numpy as np
        from MAB_algorithm import heavyTailArm

        _DRAWTIMES = 10000

        means = [i*0.05+0.65 for i in range(8)]
        arms = [heavyTailArm(i, 5, means[i], 2) for i in range(len(means))]

        for i, a in enumerate(arms):
            time0 = time.time()
            me: float = np.mean(a.draw(_DRAWTIMES))
            time1 = time.time()

            info = f"Arm with mean: {a.mean}, \
                drawn {_DRAWTIMES}, average: {me}, \
                average draw time: {(time1-time0)/_DRAWTIMES}"
            info = ' '.join(info.split())
            print(info)
            self.assertAlmostEqual(a.mean, me, delta=0.03)


if __name__ == "__main__":
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    unittest.main()
