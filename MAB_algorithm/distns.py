import numpy as np
from scipy import integrate
from scipy.special import expi
from scipy.stats import rv_continuous

__all__ = [
    "heavy_tail"
]


class heavy_tail(rv_continuous):
    """
    A heavy tail distribution with pdf that is dilated and translated from 
    `1/(x**(maxMomentOrder+1)*log(x)**2)`.

    Note:
        * The range of variable is [2-self._bias, Infinity).
        * The pdf is::

            self._coef/(x+self._bias(self._maxMomentOrder+1)*log(x+self._bias)**2).

        * The max order that the moment is finite is `maxMomentOrder`, the mean of distribution
            is `mean`.
        * :class:`heavy_tail` is a subclass of :class:`scipy.stats.rv_continuous`, so use
            :method:`rvs` to get value.
    """

    def __init__(self, maxMomentOrder, mean):
        self._maxMomentOrder = maxMomentOrder
        self._mean = mean
        self._gen_coef()
        super().__init__(a=2-self._bias, b=np.Infinity,
                         name=f"heavy_tail_{maxMomentOrder}_{mean}")

    def _gen_coef(self):
        def log_sq(x: float) -> float:
            a = np.log(x)
            return a*a
        self._coef = 1/integrate.quad(
            lambda x: 1/(np.power(x, self._maxMomentOrder+1)*log_sq(x)), 2, np.Infinity)[0]
        m = integrate.quad(
            lambda x: self._coef/(np.power(x, self._maxMomentOrder)*log_sq(x)), 2, np.Infinity)[0]
        self._bias = m-self._mean

    def _pdf(self, x, *args):
        if x < 2-self._bias:
            return 0

        def log_sq(x: float) -> float:
            a = np.log(x)
            return a*a
        return self._coef/(np.power(x+self._bias, self._maxMomentOrder+1)*log_sq(x+self._bias))

    @property
    def _variance(self):
        """Fast evaluation of this distribution"""
        if hasattr(self, "__var"):
            return self.__var
        lg2 = np.log(2)
        pow2m = np.power(2, self._maxMomentOrder)
        eim24 = (self._maxMomentOrder-2)*expi((2-self._maxMomentOrder) * lg2)\
            if self._maxMomentOrder != 2 else 0
        eimm1 = (self._maxMomentOrder-1)*expi((1-self._maxMomentOrder)*lg2)
        eimm = self._maxMomentOrder*expi(-self._maxMomentOrder*lg2)
        a = -4*eimm1+eim24-pow2m*eimm1*eimm1*lg2
        a += eimm*(4+pow2m*eim24*lg2)
        a *= pow2m*lg2
        b = 1+pow2m*eimm*lg2
        b *= b
        a /= b
        self.__var: float = a
        return a
