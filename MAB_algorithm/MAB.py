from typing import TYPE_CHECKING, Callable, List, Optional, overload

import numpy as np

if TYPE_CHECKING:
    from .arm import Arm

__all__ = [
    "MABAlgorithm",
    "DSEE",
    "robustUCB"
]


class MABAlgorithm(object):
    """
    Base class of a multi-armed bandit algorithm for inheritance.
    Multi-armed bandit - In probability theory, the multi-armed bandit problem
    (sometimes called the K- or N-armed bandit problem) is a problem in which
    a fixed limited set of resources must be allocated between competing
    (alternative) choices in a way that maximizes their expected gain,
    when each choice's properties are only partially known at the time of
    allocation, and may become better understood as time passes or by
    allocating resources to the choice. Taken from
    https://en.wikipedia.org/wiki/Multi-armed_bandit

    Note:
        * One should never construct an object by `ob = MABAlgorithm(arms)`,
            since the basic class :class:`MABAlgorithm` has not implement
            any algorithm yet.
        * To define a new algorithm, first inherits :class:`MABAlgorithm`, then
            implements method :method:`select_arm`.
        * To make Monte Carlo experiment fast and memory-conserving, one
            should define :attr:`__slots__` to limit the names of attributes,
            by adding all names of user defined attributes to :attr:`__slots__`
            before :method:`__init__` implemented.

    Args:
        arms (:obj:`List[Arm]`): Bandit arms.
    """
    __slots__ = [
        "_arms",
        "_reward_sum",
        "_counts",
        "optimal_strategy_rewards",
        "collected_rewards",
        "expected_rewards",
        "optimal_arm_rewards"
    ]

    def __init__(self, arms: List['Arm']) -> None:
        """
        Default initialization method of an Multi-armed bandit algorithm.
        """
        if not arms:
            raise ValueError("There should be at least one arm")
        self._arms = arms
        self._reset_variables()

    def _init(self, *args, **kwargs):
        """
        Initialize all user defined attributes here.
        """
        ...

    def restart(self, *args, **kwargs) -> None:
        """
        Algorithm restart. After calling this method, one can call `run_simulation()`
        again to start a new iteration.

        Override this method to restart an algorithm correctly. It is suggested that
        use `super().restart()` at the beginning of the overriding one in subclasses.
        """
        self._reset_variables()

    def _reset_variables(self):
        """
        Reset basic variables to `0.0`: :attr:`optimal_strategy_rewards`,
        :attr:`collected_rewards`, :attr:`optimal_strategy_rewards`,
        and :attr:`optimal_arm_rewards`.

        Generally one should not override this method. To reset user defined
        variables, use :method:`restart()`
        """
        self._reward_sum = np.zeros(len(self._arms))
        self._counts = np.zeros(len(self._arms))
        self.optimal_strategy_rewards = 0.0
        self.collected_rewards = 0.0
        self.expected_rewards = 0.0
        self._get_optimal_arm_rewards()

    @property
    def mean(self) -> List[float]:
        """
        Default mean estimator, using arithmetic mean.
        Should be overriden if the algorithm defines another mean estimator.
        """
        return np.array([
            self._reward_sum[i]/self._counts[i] if self._counts[i] != 0 else np.NaN for i in range(len(self._arms))
        ])

    @property
    def regret(self) -> float:
        """Represent regret of an algorithm."""
        return self.optimal_strategy_rewards - self.expected_rewards

    @overload
    def select_arm(self,
                   count: Optional[int],
                   mean_estimator: Optional[Callable[..., float]] = None,
                   *args, **kwargs
                   ) -> int:
        ...

    def select_arm(self, *args, **kwargs) -> int:
        """
        The method that returns the index of the Arm that the player selects
        on the current play. This method should be implemented in subclasses.
        Suggested arguments are listed below, and are passed to `select_arm()`
        by default.

        Args:
            count (:obj:`int`): current number of draws.
            mean_estimator (:obj:`function`, optional) gives the mean estimator
                to select arms. Generally, this function is not needed, since
                the algorithm should implement the property :property:`mean`. One
                should define this function if algorithm needs two mean estimator,
                or the arithmetic mean needs to be retained.
            other (`Any`): Any keyword argument needed. If needed, one should also
                override :method:`run_simulation()`.

        Returns:
            :obj:`int`: index of chosen arm.
        """
        raise NotImplementedError

    def _get_reward(self, index: int) -> float:
        return self._arms[index].draw()

    def _after_draw(self, iteration: int, chosen_arm_index: int, reward: float) -> None:
        """
        After process method. Should be overriden if the algorithm needs.

        Args:
            chosen_arm_index (:obj:`int`): the index of chosen arm.
            reward (:obj:`float`): the reward drawn.
        """
        ...

    def _update_current_states(self, chosen_arm_index, reward) -> None:
        self._counts[chosen_arm_index] += 1
        self._reward_sum[chosen_arm_index] += reward

    def _update_rewards_info(self, chosen_arm_index: int, reward: float):
        self.collected_rewards += reward
        self.expected_rewards += self._arms[chosen_arm_index].optimal_rewards()
        self.optimal_strategy_rewards += self.optimal_arm_rewards

    def _get_optimal_arm_rewards(self):
        optimal_arm_rewards = self._arms[0].optimal_rewards()
        for arm in self._arms:
            if optimal_arm_rewards < arm.optimal_rewards():
                optimal_arm_rewards = arm.optimal_rewards()
        self.optimal_arm_rewards = float(optimal_arm_rewards)

    def _simulation_result_dict(self, iteration: int, chosen_arm_index: int) -> dict:
        """
        `_simulation_result_dict` should at least returns these fields and values.

        Note:
            * the keys should not starts with `avg`, since Monte Carlo experiments
                use `"avg_"+key` as the 
        """
        return {
            "iteration": iteration,
            "chosen_arm": self._arms[chosen_arm_index].name,
            "regret": self.regret,
            "collected_rewards": self.collected_rewards,
            "collected_rewards_per_step": self.collected_rewards / (iteration + 1)
        }

    def run_simulation(self, number_of_iterations: int):
        """
        Run simulation and update the probabilities to pull for each arm.
        The method :method:`select_arm` should be defined in subclasses in order to call
        :method:`run_simulation` correctly. To make sure one can monitor all the variables
        in each step, this method returns an iterator.

        Note:
            * Overriding this method is not recommanded. It is recommanded that one
                overrides the subdivided methods in this method.
            * To use the result of simulation, get iterator by
                `simul = some_algorithm.run_simulation(number_of_iterations)`. Then use
                `for result in simul:` to get the result data (a dict with detailed
                information) of each iteration. Or, simply use `list(simul)` to get the
                full iteration data list.
            * To define a new result dict, override :method:`_simulation_result_dict`.
            * One can control what the algorithm would do after (or, before)
                each step by implementing :method:`_after_draw`.

        Args:
            number_of_iterations (:obj:`int`): Number of iteration steps.

        Returns:
            :obj:`Generator[dict]`: An iterator yielding meta data of each iteration.

        Yields:
            :obj:`Dict[str, float | int | str]`: Meta data of each iteration. For the
                detailed structure of this, see :method:`_simulation_result_dict`.
        """
        if number_of_iterations < 1:
            raise ValueError("Number of iterations must be positive")

        for iteration in range(number_of_iterations):
            chosen_arm_index = self.select_arm(
                count=iteration,
                mean_estimator=None
            )

            reward = self._get_reward(chosen_arm_index)

            self._update_current_states(chosen_arm_index, reward)

            self._update_rewards_info(chosen_arm_index, reward)

            yield self._simulation_result_dict(iteration, chosen_arm_index)

            self._after_draw(iteration, chosen_arm_index, reward)


class DSEE(MABAlgorithm):
    """
    Time is divided into two interleaving sequences, in one of which, arms are
    selected for exploration, and in the other, for exploitation. In the
    exploration sequence, the player playes all arms in a round-robin fashion.
    In the exploitation sequence, the player plays the arms with the largest
    sample mean.

    Args:
        arms (:obj:`List[Arm]`): Bandit arms.
        w (:obj:`float`): Parameter to define the exploration sequence.
    """
    __slots__ = [
        "_exploration",
        "__w",
        "_explore_iters",
        "explore_sum"
    ]

    def __init__(self, arms: List['Arm'], w: float) -> None:
        super().__init__(arms)
        self._init(w)

    def _init(self, w: float):
        self._exploration = [0]*len(self._arms)
        self.w = w
        self._explore_iters = self.generate_exploration(
            len(self._arms), w)
        self.explore_sum = 0

    def restart(self, w: float) -> None:
        """
        Algorithm restart with another parameter `w`.
        """
        super().restart()
        self._init(w)

    @property
    def w(self):
        """Parameter to define the exploration sequence."""
        return self.__w

    @w.setter
    def w(self, ww: float):
        if ww <= 0:
            raise ValueError("Parameter w should be positive")
        self.__w = ww

    @staticmethod
    def generate_exploration(number_of_arms: int, w: float):
        """
        DSEE exploration sequence. The iteration will not stop automately.

        Args:
            number_of_arms (:obj:`int`): The number of arms.
            w (:obj:`float`): Parameter to define the exploration sequence.
                If `number_of_exploration >= number_of_arms * ceil(w * log(n))`, then
                exploit; else, explore.
        """
        _t = 0
        _sum = 0

        while _t < number_of_arms:
            yield True
            _t += 1
            _sum += 1

        while True:
            if _sum >= number_of_arms*np.ceil(w*np.log(_t+1)):
                yield False
            else:
                yield True
                _sum += 1
            _t += 1

    def select_arm(self, *args, **kwargs) -> int:
        """
        Arm selection for DSEE.
        """
        explore = self._explore_iters.__next__()
        if explore:
            self.explore_sum += 1
            chosen_arm_index = int(np.argmin(self._exploration))
            self._exploration[chosen_arm_index] += 1
            return chosen_arm_index

        mean = self.mean
        for i in range(len(mean)):
            if mean[i] is np.NaN:
                return i

        return int(np.argmax(mean))


class robustUCB(MABAlgorithm):
    __slots__ = [
        "__v",
        "__tol",
        "reward_history",
        "_t",
        "_last_catoni_mean",
        "_psi"
    ]

    def __init__(self, arms: List['Arm'], v: float, tol: float = 1e-2, psi: Optional[Callable[..., float]] = None) -> None:
        super().__init__(arms)
        self._init(v, tol, psi)

    def _init(self, v: float, tol: float = 1e-2, psi: Optional[Callable[..., float]] = None):
        self.v = v
        self.tol = tol
        self.reward_history: List[List[float]] = [
            [] for _ in range(len(self._arms))
        ]
        self._t = 1
        self._last_catoni_mean = np.zeros(len(self._arms))
        self._psi = psi

    def restart(self, v: float, tol: float = 1e-2, psi: Optional[Callable[..., float]] = None) -> None:
        super().restart()
        self._init(v, tol, psi)

    @property
    def v(self) -> float:
        return self.__v

    @v.setter
    def v(self, vv: float):
        if vv <= 0:
            raise ValueError("Parameter v should be positive")
        self.__v = vv

    @property
    def tol(self) -> float:
        return self.__tol

    @tol.setter
    def tol(self, tt: float):
        if tt <= 0 or tt > 1:
            raise ValueError(
                "The tolerance should neither be negative nor too big")
        self.__tol = tt

    def alpha(self, delta: float, n: int):
        lg1divd = np.log(1/delta)
        v = self.v
        return np.sqrt(2*lg1divd/(n*(v+2*v*lg1divd/(n-2*lg1divd))))

    @staticmethod
    def psi(x: float) -> float:
        if x <= 1 and x >= -1:
            return x-x*x*x/6
        if x > 1:
            return np.log(x)+5/6
        return -np.log(-x)-5/6

    def get_Catoni_mean(self, index: int) -> float:
        t = self._t
        delta = 1/(t*t)
        n = len(self.reward_history[index])

        guess0 = self._last_catoni_mean[index]
        a = self._sum_psi(index, guess0, delta, n)

        if a > 0:
            while a > 0:
                guess0 += 1
                a = self._sum_psi(index, guess0, delta, n)
            ans = self._find_root(index, delta, n, guess0-1, guess0)
        else:
            while a < 0:
                guess0 -= 1
                a = self._sum_psi(index, guess0, delta, n)
            ans = self._find_root(index, delta, n, guess0, guess0+1)

        self._last_catoni_mean[index] = ans
        return ans

    def _find_root(self, index: int, delta: float, n: int, a: float, b: float) -> float:
        tol = self.tol
        while b - a > tol:
            guess = (b+a)/2
            if self._sum_psi(index, guess, delta, n) > 0:
                a = guess
            else:
                b = guess
        return (a+b)/2

    def _sum_psi(self, index: int, guess: float, delta, n) -> float:
        alpha_d = self.alpha(delta, n)
        if self._psi:
            return sum(self._psi(alpha_d*(x-guess)) for x in self.reward_history[index])
        return sum(self.psi(alpha_d*(x-guess)) for x in self.reward_history[index])

    @property
    def mean(self) -> List[float]:
        if self._t <= len(self._arms):
            ans = np.array(
                [x[0] if x else np.NaN for x in self.reward_history]
            )
            return ans

        ans = np.zeros(len(self._arms))
        t = self._t

        for i in range(len(self._arms)):
            s = len(self.reward_history[i])
            ans[i] = self.get_Catoni_mean(i)+2*np.sqrt(2*self.v*np.log(t)/s)
        return ans

    def select_arm(self, *args, **kwargs) -> int:
        lgt_8 = 8*np.log(self._t)
        for i in range(len(self._arms)):
            if len(self.reward_history[i]) < lgt_8:
                return i

        return np.argmax(self.mean)

    def _after_draw(self, iteration: int, chosen_arm_index: int, reward: float) -> None:
        self._t += 1
        self.reward_history[chosen_arm_index].append(reward)
