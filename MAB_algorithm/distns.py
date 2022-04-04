import logging

import numpy as np
from scipy import integrate
from scipy.special import expi
from scipy.stats import rv_continuous

from MAB_algorithm.mabCutils import heavytail_dist_pdf
from MAB_algorithm.MAButils import NewtonIteration

__all__ = [
    "heavy_tail",
    "paretoTypeIIDist"
]


class heavy_tail(rv_continuous):
    """
    A heavy tail distribution with pdf that is dilated and translated from 
    `1/(x**(maxMomentOrder+1)*log(x)**2)`. Input mean value `mean` and 97 percent quantile `b` 
    to generate a heavy tail distribution.

    Note:
        * Do not input a very large `maxMomentOrder`, because it may cause precision problem.
            Less than 10 is suggested. 
        * `mean` being less than `b` is suggested.
        * The range of variable is `[(2-_beta)/_alpha, +Inf)`. _alpha and _beta is 
            evaluated by mean and `b`.
        * The pdf (where k is a number to make the integral on real line equal to 1) is::

            k/((_alpha*x+_beta)**(_maxMomentOrder+1) * log(_alpha*x+_beta)**2).

        * :class:`heavy_tail` is a subclass of :class:`scipy.stats.rv_continuous`, so use
            :method:`rvs` to get value.

    Args:
        maxMomentOrder (:obj:`float`): Max order of finite moment.
        mean (:obj:`float`): Mean of distribution.
        b (:obj:`float`): The 97 percent quantile.
    """

    def __init__(self, maxMomentOrder, mean, b):
        """Store parameters and generate coefficients."""
        self._maxMomentOrder = maxMomentOrder
        self._mean = mean
        self._b = b
        self._gen_coef()
        super().__init__(name=f"heavy_tail_{maxMomentOrder}_{mean}_{b}")

    def _gen_coef(self):
        def log_sq(x: float) -> float:
            a = np.log(x)
            return a*a

        lg2 = np.log(2)
        m = self._maxMomentOrder
        m22 = np.power(2, m)*lg2
        eim2 = expi(-m*lg2)
        if np.abs(eim2) < 1e-6:
            logging.getLogger(__name__).warning(
                "maxMomentOrder is too large. This may cause precision problem.")
        self._coef = 1/integrate.quad(
            lambda x: 1/(np.power(x, self._maxMomentOrder+1)*log_sq(x)), 2, np.Infinity)[0]

        me = integrate.quad(
            lambda x: self._coef/(np.power(x, self._maxMomentOrder)*log_sq(x)), 2, np.Infinity)[0]

        def _f(x: float) -> float:
            lgx = np.log(x)
            return -0.03+m22*(np.power(x, -m)+m*expi(-m*lgx)*lgx)/(lgx+m22*m*eim2*lgx)

        def _df(x: float) -> float:
            lgx = np.log(x)
            return -m22*np.power(x, -m)/(x*lgx*lgx*(1+m*m22*eim2))

        x0 = 2.5
        while _f(x0) > 0:
            x0 += 1
        x0 -= 1

        def _tol(x) -> float:
            _a = _df(x)
            if _a < 0:
                _a = -_a
            if _a <= 0.01:
                return _a*1e-14
            return 1e-15

        r = NewtonIteration.iter_full(_f, _df, x0, _tol, 50)
        if r < me:
            raise ValueError("maxMomentOrder is too large")

        self._alpha = (r-me)/(self._b-self._mean)

        self._beta = r-self._alpha*self._b

    def _pdf(self, x, *args):
        return heavytail_dist_pdf(self._alpha, self._beta, self._coef, self._maxMomentOrder, x)

    @property
    def _variance(self):
        """Fast evaluation of the variance."""
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
        a /= self._alpha*self._alpha
        self.__var: float = a
        return a


class paretoTypeIIDist(rv_continuous):
    """
    A Pareto distribution (type II from wikipedia) with pdf 
    `alpha/sigma * (1+(x-mu)/sigma)**(-1-alpha)`.

    Note:
        * Pareto distribution type II (hereinafter referred to as Pareto distribution) is 
            heavy-tailed but easy to compute.
        * Pareto distribution doesn't have finite alpha-order moment. However, for any 
            `p<alpha`, p-th order moment exists.
        * `mu < sigma` should hold.
        * The mean of Pareto distribution is not parameter `mu`. Mean is `mu+sigma/(alpha-1)`.
        * Pareto distribtion has support `x>mu`.
        * :class:`paretoTypeIIDist` is a subclass of :class:`scipy.stats.rv_continuous`, 
            so use :method:`rvs` to get value.
        * For Pareto distribution, the q-th central moment `vq` and p-th central moment 
            `vp` s.t. `vp = C * vq**(p/q)`, where `C = C(p, q, alpha)` is not related to 
            `mu` and `sigma`.

    Args:
        maxMomentOrder (:obj:`float`): The superior of finite moment order.
        mu (:obj:`float`): Left bound of the support.
        sigma (:obj:`float`): A parameter related to moment.
    """

    def __init__(self, alpha: float, mu: float, sigma: float):
        """Store parameters and generate coefficients."""
        if mu >= sigma:
            raise ValueError(
                "Invalid parameters for Pareto distribution (type II)")
        self._alpha = alpha
        self._mu = mu
        self._sigma = sigma

        super().__init__(name=f"pareto_typeII_{alpha}_{mu}_{sigma}")

    def _pdf(self, x, *args):
        return self._alpha/self._sigma * np.power(1+(x-self._mu)/self._sigma, -1-self._alpha) if x > self._mu else 0

    @property
    def _mean(self):
        return self._mu + self._sigma/(self._alpha-1)

    @property
    def _variance(self):
        """Fast evaluation of the variance."""
        return self._alpha*self._sigma*self._sigma / ((self._alpha-1)*(self._alpha-1)*(self._alpha-2))
