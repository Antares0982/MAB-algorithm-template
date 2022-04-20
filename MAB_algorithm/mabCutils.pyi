"""
Cython classes/functions library of MAB_algorithm.

These classes/functions are used to accelerate evaluations.

Note:
    * To use the classes/functions, type of (initialize) arguments have to match 
        with the type hints provided.
"""
from typing import Tuple


class mabarray(object):
    """
    The C++ `double[]` (python: `List[float]`) array wrapper class, 
    with very high speed operation capability. 
    At initialize, one should explicitly determine the `maxsize` of the array, 
    then `new double[maxsize]` is declared. The calculation speed will be about 
    200 times faster than pure python style calculations.

    Note:
        * It is the basic data structure for MAB C++ functions like :function:`getCatoniMean()`, 
            and doesn't support numpy algorithms such as :function:`np.argmax()`, :function:`np.mean()`.
        * Use :method:`add()` to append element at the end of the array 
            (with time complexity `O(1)`).
        * :method:`avg()` gives the average of all elements.
        * you can treat the mabarray as normal `List[float]` object, but python iterator
            operation is not supported for now. (python iteration is slow and not recommended.
            Also, applying python function to each element of `mabarray` is not recommended.)
    """

    def __init__(self, maxsize: int) -> None: ...
    def add(self, v: float) -> None: ...
    def avg(self) -> float: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> float: ...
    def __setitem__(self, index: int, value: float): ...


class medianOfMeanArray(object):
    """
    The wrapper class of C++ class `medianOfMeanArray`. `medianOfMeanArray` is
    a subclass of `mabarray` in C++; it provides all functionality of mabarray, 
    and also a fast way to compute median of means.

    Note:
        * `medianOfMeanArray` also stores presum and means, which will 
            cost much more space than `mabarray`.
        * :method:`medianMean()` has an average time cost O(log n).
    """

    def __init__(self, maxsize: int) -> None: ...
    def add(self, v: float) -> None: ...
    def avg(self) -> float: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> float: ...
    def __setitem__(self, index: int, value: float): ...
    def medianMean(self, binsizeN: int) -> float: ...


def getCatoniMean(
    v: float,
    iteration: int,
    guess: float,
    arr: mabarray,
    tol: float
) -> Tuple[float, int]:
    """
    C++ Catoni mean fast evaluation function.

    Catoni mean requires finding root of `f(x) = sum(psi(alpha*(x-arr[i])))`, which is 
    very slow in python. `getCatoniMean` is about 140 times faster than pure python implement.
    """
    ...


def getTruncatedMean(
    u: float,
    ve: float,
    iteration: int,
    arr: mabarray
) -> float:
    """
    C++ robust UCB trucnated mean fast evaluation function.

    Truncated robust UCB requires sum of reward history whose index satisfies some condition, 
    which costs much time.
    """
    ...


def getMedianMean(
    v: float,
    ve: float,
    iteration: int,
    arr: mabarray
) -> float:
    """
    C++ robust UCB median mean fast evaluation function.

    Median robust UCB requires sum of reward history whose index satisfies some condition, 
    which costs much time.
    """
    ...


def heavytail_dist_pdf(
    alpha: float,
    beta: float,
    coef: float,
    maxmomentorder: float,
    x: float
) -> float:
    """
    Cython heavy-tailed distribution's (defined in `MAButils.py`) pdf.
    Each time a heavy-tailed distribution is drawn, the pdf of the distribution is called many times.
    This C-style function can be used to accelerate the process.
    """
    ...
