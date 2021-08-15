from typing import Callable, Iterator, List, Optional

from .arm import *


class MABAlgorithm(object):
    """
    Multi-armed bandit - In probability theory, the multi-armed bandit problem (sometimes called the K- or N-armed
    bandit problem) is a problem in which a fixed limited set of resources must be allocated between competing
    (alternative) choices in a way that maximizes their expected gain, when each choice's properties are only partially
    known at the time of allocation, and may become better understood as time passes or by allocating resources to the
    choice. Taken from https://en.wikipedia.org/wiki/Multi-armed_bandit
    """

    def __init__(self, arms: List[Arm]) -> None:
        """
        Default initialization method of an Multi-armed bandit algorithm.

        Args:
            arms (:obj:`List[Arm]`): Bandit arms.
        """
        if not arms:
            raise ValueError("There should be at least one arm")
        self._arms = arms
        self._reward_sum = np.zeros(len(self._arms))
        self._counts = np.zeros(len(self._arms))

    @property
    def mean(self) -> List[float]:
        """
        Default mean estimator, using arithmetic mean.
        Should be overriden if the algorithm defines another mean estimator.
        """
        return [self._reward_sum[i]/self._counts[i] if self._counts[i] != 0 else np.NaN for i in range(len(self._arms))]

    @overload
    def select_arm(self,
                   count: Optional[int],
                   mean_estimator: Optional[Callable[..., float]] = None,
                   *args, **kwargs
                   ) -> int:
        ...

    def select_arm(self, *args, **kwargs) -> int:
        """
        The method that returns the index of the Arm that the player selects on the current play.
        This method should be implemented in subclasses.
        Suggested arguments are listed below, and are passed to `select_arm()` by default.

        Args:
            count (:obj:`int`): current number of draws.
            mean_estimator (:obj:`function`, optional) gives the mean estimator to select arms.
                Generally, this function is not needed, since the algorithm should
                implement the property :attr:`mean`. One should define this function
                if algorithm needs two mean estimator, or the arithmetic mean needs to be
                retained.
            other (`Any`): Any keyword argument needed. If needed, one should also
                override :attr:`run_simulation()`.

        Returns:
            :obj:`int`: index of chosen arm.
        """
        raise NotImplementedError

    def _get_reward(self, index: int) -> float:
        return self._arms[index].draw()

    def _after_draw(self,  chosen_arm_index: int, reward: float) -> None:
        """
        After process method. Should be overriden if the algorithm needs.

        Args:
            chosen_arm_index (:obj:`int`): the index of chosen arm.
            reward (:obj:`float`): the reward drawn.
        """
        pass

    def _update_current_states(self, chosen_arm_index, reward) -> None:
        self._counts[chosen_arm_index] += 1
        self._reward_sum[chosen_arm_index] += reward

    def run_simulation(self, number_of_iterations: int) -> Iterator[dict]:
        """
        Run simulation and update the probabilities to pull for each arm.
        The method `select_arm()` should be defined in subclasses in order to call `run_simulation` correctly.

        Args:
            number_of_iterations (:obj:`int`): Number of number_of_iterations.

        Returns:
            :obj:`Iterator[dict]`: Meta data of each iteration.
        """
        if number_of_iterations < 1:
            raise ValueError("Number of iterations must be positive")

        number_of_arms = len(self._arms)

        optimal_strategy_rewards = 0.0
        collected_rewards = 0.0

        optimal_arm_rewards = self._arms[0].optimal_rewards()
        for arm_index in range(1, number_of_arms):
            if optimal_arm_rewards < self._arms[arm_index].optimal_rewards():
                optimal_arm_rewards = self._arms[arm_index].optimal_rewards()

        expected_rewards = 0.0

        # rewards = np.zeros([number_of_arms, self._iterations])

        # for arm_index in range(0, number_of_arms):
        #     rewards[arm_index] = self._arms[arm_index].draw(self._iterations)

        for iteration in range(0, number_of_iterations):
            chosen_arm_index = self.select_arm(
                count=iteration,
                mean_estimator=None
            )
            reward = self._get_reward(chosen_arm_index)

            self._update_current_states(chosen_arm_index, reward)

            collected_rewards += reward
            expected_rewards += self._arms[chosen_arm_index].optimal_rewards()
            #optimal_strategy_rewards += np.max(rewards[:, iteration])
            optimal_strategy_rewards += optimal_arm_rewards
            # regret = optimal_strategy_rewards - collected_rewards
            regret = optimal_strategy_rewards - expected_rewards

            ans = {"iteration": iteration, "chosen_arm": self._arms[chosen_arm_index].name,
                   "regret": regret, "collected_rewards": collected_rewards,
                   "avg_regret": regret / (iteration + 1),
                   "avg_collected_rewards": collected_rewards / (iteration + 1)}
            yield ans
            self._after_draw(chosen_arm_index, reward)

    def restart(self, *args, **kwargs) -> None:
        """
        Algorithm restart. After calling this method, one can call `run_simulation()` again
        to start a new iteration.
        Override this method to restart an algorithm correctly.
        """
        raise NotImplementedError


class DSEE(MABAlgorithm):
    """
    Time is divided into two interleaving sequences, in one of which, arms are selected for exploration, 
    and in the other, for exploitation. In the exploration sequence, the player playes all arms in a round-robin fashion.
    In the exploitation sequence, the player plays the arms with the largest sample mean.

    Args:
        arms (:obj:`List[Arm]`): Bandit arms.
        w (:obj:`float`): Parameter to define the exploration sequence.
    """

    def __init__(self, arms: List[Arm], w: float) -> None:
        super().__init__(arms)
        self._exploration = [0]*len(arms)
        self.__w = w
        self._explore_iters: Iterator[bool] = self.generate_exploration(
            len(arms), w)
        self.explore_sum = 0

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
    def generate_exploration(number_of_arms: int, w: float) -> Iterator[bool]:
        """
        DSEE exploration sequence. The iteration will not stop automately. 

        Args:
            number_of_arms (:obj:`int`): The number of arms.
            w (:obj:`float`): Parameter to define the exploration sequence.
                If `number_of_exploration >= number_of_arms * ceil(w * log(n))`, then exploit;
                else explore.
        """
        _count = 0
        _sum = 0

        while _count < number_of_arms:
            yield True
            _count += 1
            _sum += 1

        while True:
            if _sum >= number_of_arms*np.ceil(w*np.log(_count+1)):
                yield False
            else:
                yield True
                _sum += 1
            _count += 1

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

    def restart(self, w: float) -> None:
        """
        Algorithm restart with another parameter `w`.
        """
        self.w = w
        self._explore_iters = self.generate_exploration(len(self._arms), w)
        self.explore_sum = 0
