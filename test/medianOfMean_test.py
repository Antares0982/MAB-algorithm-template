import os
import sys
import unittest


class MedianOfMean_test(unittest.TestCase):
    def test_mean(self):
        import time
        from random import randint

        import numpy as np
        from MAB_algorithm.mabCutils import (calculateMedianMean, mabarray,
                                             medianOfMeanArray)

        LENGTH = 100000

        arr1 = mabarray(LENGTH)
        arr2 = medianOfMeanArray(LENGTH)
        totaltime1: float = 0.
        totaltime2: float = 0.

        for i in range(LENGTH):
            dum = randint(0, 99)/100  # generate random float from 0.00 to 0.99
            # add a random double into array
            arr1.add(dum)
            arr2.add(dum)

            bins = np.max([1, np.floor(
                np.min([1+16*np.log(i+1), len(arr1)/2]))]).astype(np.int64)
            # compute time for simple medianOfMean
            time0 = time.time()
            m1 = calculateMedianMean(arr1, bins)
            time1 = time.time()
            totaltime1 += time1-time0

            # compute time for medianOfMeanArray
            time0 = time.time()
            m2 = arr2.medianMean(bins)
            time1 = time.time()
            totaltime2 += time1-time0

            self.assertAlmostEqual(m1, m2, delta=0.00001, msg=f"not equal!")

        info = f"test passed, simple median of mean consumed {totaltime1} secs, medianOfMeanArray consumed {totaltime2} secs."
        print(info)


if __name__ == "__main__":
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    unittest.main()
