import os
import sys
import unittest


class medianOfMean_test(unittest.TestCase):
    def test_mean(self):
        import time
        from random import randint

        import numpy as np
        from MAB_algorithm.mabCutils import (getMedianMean, mabarray,
                                             medianOfMeanArray)

        LENGTH = 10000

        arr1 = mabarray(LENGTH)
        arr2 = medianOfMeanArray(LENGTH)
        totaltime1:float = 0.
        totaltime2:float = 0.

        def printarr(arr)->str:
            s = ""
            sep = ""
            for i in arr:
                s+=sep
                s+=str(i)
                sep=", "
            return s

        for i in range(LENGTH):
            fake_itercount = max(1, int(i+np.log(i+1)))
            dum = randint(0,99)/100 # generate random float from 0.00 to 0.99
            # add a random double into array
            arr1.add(dum)
            arr2.add(dum)

            # compute time for simple medianOfMean
            time0 = time.time()
            m1 = getMedianMean(0.25,1,fake_itercount,arr1)
            time1= time.time()
            totaltime1+=time1-time0

            # compute time for medianOfMeanArray
            time0 = time.time()
            k = max(int(min(1 + 16 * np.log(fake_itercount,dtype=np.float32), (i+1)/2)), 1)
            binsizeN = (i+1)//k
            m2 = arr2.medianMean(binsizeN)
            time1= time.time()
            totaltime2+=time1-time0

            self.assertAlmostEqual(m1, m2, msg=f"not equal, {printarr(arr1)}")

        info = f"test passed, simple median of mean consumed {totaltime1} secs, \
            medianOfMeanArray consumed {totaltime2} secs."
        print(info)


if __name__ == "__main__":
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    unittest.main()
