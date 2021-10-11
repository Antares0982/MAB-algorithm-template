from typing import List, Tuple, Union, overload

import numpy as np
from scipy.stats._continuous_distns import truncnorm
from scipy.stats._discrete_distns import bernoulli

from .distns import *

__all__ = [
    "Arm",
    "TruncNormArm",
    "BernoulliArm",
    "heavyTailArm",
    "armList"
]


class Arm(object):
    def __init__(self, name: Union[str, int]) -> None:
        """Base class of an arm of bandit for inheritance."""
        self.__name = name

    @property
    def name(self):
        """
        Name of the arm.
        The name is defined in __init__() and should not be changed.
        """
        return self.__name

    def _get_rewards(self, size: int) -> np.ndarray:
        """Get rewards according to a certain size.

        Args:
            size (:obj:`int`): number of draws.

        Returns:
            :obj:`numpy.ndarray`: List of rewards.

        This method is intended for overriden by subclasses and should not be called
        externally.
        """
        raise NotImplementedError

    @overload
    def draw(self) -> float:
        ...

    @overload
    def draw(self, size: int) -> np.ndarray:
        ...

    def draw(self, size=None):
        """Get rewards.

        Args:
            size (:obj:`int`, optional): Number of draws. If is `None`, then draw once
                and returns a float. Else the method returns a list of rewards of each
                draw.

        Returns:
            :obj:`numpy.ndarray` | :obj:`int`: Reward(s).
        """
        if size is None:
            return self._get_rewards(1)[0]
        return self._get_rewards(size)

    def optimal_rewards(self) -> float:
        raise NotImplementedError

    def variance(self) -> float:
        raise NotImplementedError

    def get_dict(self) -> dict:
        """Return the arm values as dictionary with name and probability."""
        return {"name": self.name}


class TruncNormArm(Arm):
    """
    An arm with truncacted normal distribution.

    Args:
        mu (:obj:`float`): mean of the normal distribution.
        sigma (:obj:`float`): standard deviation of the normal distribution.
    """

    def __init__(self, name: Union[str, int], mu: float, sigma: float) -> None:
        if sigma <= 0:
            raise ValueError("Sigma should be positive")
        super().__init__(name)
        self.__mu = mu
        self.__sigma = sigma

    @property
    def mu(self) -> float:
        return self.__mu

    @property
    def sigma(self) -> float:
        return self.__sigma

    def optimal_rewards(self) -> float:
        return truncnorm.stats((0 - self.__mu) / self.__sigma,
                               (1 - self.__mu) / self.__sigma,
                               self.__mu,
                               self.__sigma,
                               moments="m")

    def variance(self) -> float:
        return truncnorm.stats((0 - self.__mu) / self.__sigma,
                               (1 - self.__mu) / self.__sigma,
                               self.__mu,
                               self.__sigma,
                               moments="v")

    def _get_rewards(self, size: int) -> np.ndarray:
        return truncnorm.rvs((0 - self.__mu) / self.__sigma,
                             (1 - self.__mu) / self.__sigma,
                             self.__mu,
                             self.__sigma,
                             size=size)


class BernoulliArm(Arm):
    """
    An arm with Bernoulli distribution.

    Args:
        p (:obj:`float`): The possibility that the random variable is 1, else 0.
    """

    def __init__(self, name: Union[str, int], p: float) -> None:
        if p < 0 or p > 1:
            raise ValueError("p should be between 0 and 1")
        super().__init__(name)
        self.__p = p

    @property
    def p(self) -> float:
        return self.__p

    def optimal_rewards(self) -> float:
        return self.p

    def variance(self) -> float:
        return self.p*(1-self.p)

    def _get_rewards(self, size: int) -> np.ndarray:
        return bernoulli.rvs(self.p, size=size)


class heavyTailArm(Arm):
    """
    An arm with heavy tailed distribution, using translation of
    `1/(x**(maxMomentOrder+1)*log(x)**2)` as pdf.

    Note:
        * The variance (or, moments) only depends on `maxMomentOrder`.
        * For the case `maxMomentOrder < 2`, the distribution does not have finite variance.

    Args:
        maxMomentOrder (:ob:`float`): `maxMomentOrder` is the max order of finite moments. If
            s>maxMomentOrder, the s-order moment is infinity. 
        mean (:obj:`float`): The mean of distribution. Translate
            `1/(x**(maxMomentOrder+1)*log(x)**2)` to make the mean of distribution equal `mean`.
    """

    def __init__(self, name: Union[str, int], maxMomentOrder: float, mean: float, mainbound: float) -> None:
        super().__init__(name)
        if maxMomentOrder < 1:
            raise ValueError("Mean of random variable must exist")
        self.__maxMomentOrder = maxMomentOrder
        self.__mean = mean
        self.__mainbound = mainbound
        self._heavy_tail_random_var_gen = heavy_tail(
            maxMomentOrder, mean, mainbound)

    @property
    def maxMomentOrder(self) -> float:
        return self.__maxMomentOrder

    @property
    def mean(self) -> float:
        return self.__mean

    @property
    def mainbound(self) -> float:
        return self.__mainbound

    def _get_rewards(self, size: int) -> np.ndarray:
        return self._heavy_tail_random_var_gen.rvs(size=size)

    def optimal_rewards(self) -> float:
        return self.mean

    def variance(self) -> float:
        """
        The variance can be evaluated directly and accurately.
        """
        if self.maxMomentOrder < 2:
            return np.Infinity
        return self._heavy_tail_random_var_gen._variance

    def moment(self, m: float):
        if m > self.maxMomentOrder:
            return np.Infinity
        return 1  # TODO(Antares): ...

    @overload
    def draw(self) -> float:
        ...

    @overload
    def draw(self, size: int) -> np.ndarray:
        ...

    def draw(self, size=None):
        if size is None:
            while True:
                try:
                    ans = super().draw()
                except Exception:
                    continue
                return ans
        else:
            ans = np.array([0.0 for i in range(size)])
            for i in range(size):
                ans[i] = self.draw()
            return ans


class armList(object):
    """
    This class only contains functions that useful to a list of `Arm` objects.
    Should not initialize an instance of this class.
    """
    @staticmethod
    def get_optimal_arm_index(arms: List[Arm]) -> int:
        if not arms:
            raise ValueError("There is no arm.")

        return np.argmax(x.optimal_rewards() for x in arms)

    @staticmethod
    def get_optimal_arm_rewards(arms: List[Arm]) -> int:
        ind = armList.get_optimal_arm_index(arms)
        return arms[ind].optimal_rewards()

    @staticmethod
    def get_optimal_arm_index_and_rewards(arms: List[Arm]) -> Tuple[int, float]:
        ind = armList.get_optimal_arm_index(arms)
        return ind, arms[ind].optimal_rewards()

    @staticmethod
    def get_nth_arm_index_and_rewards(arms: List[Arm], *indexes: int) -> Tuple[Tuple[int, float], ...]:
        """
        Returns the n-th optimal arm index and rewards.

        Args:
            arms (:obj:`List[Arm]`): List of `Arms` object.
            indexes (:obj:`Tuple[int]`): One index or indexes of arms sorted by optimal
                rewards.

        Returns:
            `Tuple[Tuple[int, float], ...]`: Tuple of `(index, rewards)`.

        Raises:
            ValueError: If number of arms is less than 2, or the index argument is
                over bounded.
        """
        if len(arms) < 2:
            raise ValueError("Should have at least 2 arms.")
        for arg in indexes:
            if arg >= len(arms):
                raise ValueError(
                    "Index of sorted list should not be greater than length of arms")
        lst = [(i, float(arms[i].optimal_rewards())) for i in range(len(arms))]
        lst.sort(key=lambda x: x[1], reverse=True)

        return tuple(lst[x] for x in indexes)

    @staticmethod
    def get_nth_arm_index(arms: List[Arm], *indexes: int) -> Tuple[int]:
        """
        Returns the n-th optimal arm index.

        Args:
            arms (:obj:`List[Arm]`): List of `Arms` object.
            indexes (:obj:`Tuple[int]`): One index or indexes of arms sorted by optimal
                rewards.

        Returns:
            `Tuple[int]`: Tuple of `index`.

        Raises:
            ValueError: If number of arms is less than 2, or the index argument is
                over bounded.
        """
        dum = armList.get_nth_arm_index_and_rewards(arms, *indexes)
        return tuple(x[0] for x in dum)

    @staticmethod
    def get_nth_suboptimal_arm_rewards(arms: List[Arm], *indexes: int) -> Tuple[float]:
        dum = armList.get_nth_arm_index_and_rewards(arms, *indexes)
        return tuple(x[0] for x in dum)

    @staticmethod
    def getmaxVariance(arms: List[Arm]) -> float:
        return np.max([x.variance() for x in arms])
