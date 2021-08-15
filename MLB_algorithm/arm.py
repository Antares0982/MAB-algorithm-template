from typing import Union, overload

import numpy as np
from scipy.stats._continuous_distns import truncnorm
from scipy.stats._discrete_distns import bernoulli


class Arm(object):
    def __init__(self, name: Union[str, int]) -> None:
        """Base class for an arm."""
        self.__name = name

    @property
    def name(self):
        """Name of the arm.
        The name is defined in __init__() and should not be changed.
        """
        return self.__name

    def _get_rewards(self, size: int) -> np.ndarray:
        """Get rewards according to a certain size.

        Args:
            size (:obj:`int`): number of draws.

        Returns:
            :obj:`numpy.ndarray`: List of rewards.

        This method is intended for override by subclasses and should not be called externally.
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
            size (:obj:`int`, optional): Number of draws. If is `None`, then draw once and returns a float.
                Else the method returns a list of rewards of each draw.

        Returns:
            :obj:`numpy.ndarray` | :obj:`int`: Reward(s).
        """
        if size is None:
            return self._get_rewards(1)[0]
        return self._get_rewards(size)

    def optimal_rewards(self) -> float:
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

    def _get_rewards(self, size: int) -> np.ndarray:
        """
        Extracts rewards from Normal distribution. SUPPORT ON [0,1]
        :param size: number of draws.
        :return: rewards.
        """
        return truncnorm.rvs((0 - self.__mu) / self.__sigma,
                             (1 - self.__mu) / self.__sigma,
                             self.__mu,
                             self.__sigma,
                             size=size)


class BernoulliArm(Arm):
    """
    An arm with Bernoulli distribution.
    """

    def __init__(self, name: Union[str, int], p: float) -> None:
        super().__init__(name)
        self.__p = p

    @property
    def p(self) -> float:
        return self.__p

    def optimal_rewards(self) -> float:
        return self.p

    def _get_rewards(self, size: int) -> np.ndarray:
        """
        Extracts rewards from Bernoulli distribution.
        """
        return bernoulli.rvs(self.p, size=size)
