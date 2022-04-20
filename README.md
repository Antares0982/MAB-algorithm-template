# MAB-algorithm

![image](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)![image](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)![image](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)![image](https://img.shields.io/github/license/Antares0982/MAB-algorithm-template)

[Code template for Multi-Armed Bandit algorithms](https://github.com/Antares0982/MAB-algorithm-template)

### MAB model

In probability theory and machine learning, the **multi-armed bandit problem** is a problem in which a fixed limited set of resources must be allocated between competing (alternative) choices in a way that maximizes their expected gain, when each choice's properties are only partially known at the time of allocation, and may become better understood as time passes or by allocating resources to the choice. This is a classic reinforcement learning problem that exemplifies the exploration–exploitation tradeoff dilemma. 

### How To Use

Requiring python3.

```python
pip3 install MAB-algorithm
```

### A Brief Introduction

##### Arms

An MAB model includes a list of arms. This template has defined several arms with different distributions, for example, to define 10 arms with truncated normal distribution:

```python
from MAB_algorithm import *
armlist = [TruncNormArm(name=i, mu=i, sigma=1) for i in range(10)]
```

The information of pre-defined arms and how to can be found in the arms' docstring.

This template has also defined an easy-to-use heavy-tailed distribution, which is considered frequently in MAB problems.

The class `armList` has some useful functions for an arm list.

To define a new type of arm, inherit base class `Arm`. You need to implement these methods:

```python
import numpy as np
from MAB_algorithm import *

class userDefinedArm(Arm):
    def __init__(self, name, *args, **kwargs):
        # Change the initialization parameters for your own purpose, but keep `name` aside.
        super().__init__(name)
        # Implements goes here.
        ...

    def _get_rewards(self, size: int) -> np.ndarray:
        """Return the list of rewards drawn from user defined distribution
        when the arm was played `size` times"""
        # Implements goes here.
        ...

    def optimal_rewards(self) -> float:
        """Return the mean of the reward distribution."""
        # Implements goes here.
        ...

   	def variance(self) -> float:
        """Return the variance of the reward distribution. 
        If you don't need variance, just ignore this."""
        # Implements goes here.
        ...
```

Note: for how to define a customized distribution, see [Python: how to define customized distributions?](https://stackoverflow.com/questions/46055690/python-how-to-define-customized-distributions)

##### Algorithms

An MAB algorithm is a strategy to choose arm in order to gain best rewards. To define an algorithm, inherit `MABAlgorithm` and implement `select_arm`. The needed information is listed in the docstring.

```python
import numpy as np
from MAB_algorithm import *

class SimpleMAB(MABAlgorithm):
    def select_arm(self, count: int, *args, **kwargs) -> int:
        if count < len(self._arms):
            return count
        return np.argmax(self.mean) # play the one with best average reward history
```

It is easy to implement your algorithm by extending the methods pre-defined. Always feel free to use the pre-defined attributes. `_reward_sum[i]` denotes the sum of reward of arm `i`, `_counts[i]` denotes the number of times that arm is played, `optimal_strategy_rewards` denotes the expected reward of the ideal case that arm distribtutions are known (i.e. always choose the arm with best reward), `collected_rewards` denotes the collected rewards, `expected_rewards` denotes the expected total reward under this strategy, and `optimal_arm_rewards` denotes the mean of reward distribution for the best arm.

The pre-defined `mean` attribute is arthmetic mean of reward history, and it can be modified to your own mean estimator, which can be used in your `select_arm`.

```python
import numpy as np
from MAB_algorithm import *

class MyAlgorithm(MABAlgorithm):
    """
    Use `mean[i]+(t**0.25)/(s[i]**0.25)` as mean estimator,
    where `t` is time step, and `s[i]` is the number of times arm `i` was played.
    Note: this algorithm may have linear order of regret for heavy-tailed
    arm when max order of finite moment is 4.
    """
    @property
    def mean(self) -> List[float]:
        mean = super().mean
        total = sum(self._counts)
        for i in range(len(self._arms)):
            mean[i] += np.power(total, 0.25) / np.power(self._counts[i], 0.25)
        return mean

    def select_arm(self, count: int, *args, **kwargs) -> int:
        if count < len(self._arms):
            return count
        return np.argmax(self.mean)
```

Also, pre-defined `regret` is the regret under the current strategy.

For how to define a complicated algorithm, please see the implementation of `robustUCB` in the source code. If you want to change the base logic of MAB algorithm, override `_update_current_states`, `_update_rewards_info`, `_after_draw`, and if you want more information from output, override `_simulation_result`. This template also provides functions for Newton method to find root.

##### Simulations

To start your simulation, use method `run_simulation` for iterator mode:

```python
from MAB_algorithm import *
armlist = [TruncNormArm(name=i, mu=i, sigma=1) for i in range(10)]
dsee = DSEE(arms=armlist, w=10)
dseegenerator = dsee.run_simulation(10000) # T=10000
i = 0
for ans in dseegenerator:
    # `ans` will be a dict containing the information of algorithm at time step i.
    # Iterator is suggested, since simulation will be time-consuming,
    # and one can store every information needed at each step using iterator.
    print(ans)
    i += 1
```

Or, use `run_simulation_tolist` to get all information directly across the input time steps.

```python
from MAB_algorithm import *
armlist = [TruncNormArm(name=i, mu=i, sigma=1) for i in range(10)]
dsee = DSEE(arms=armlist, w=10)
allinfo = dsee.run_simulation_tolist(10000) # T=10000.
# This step is time-consuming and may give no log during the whole time.
# `allinfo` will be a list of 10000 dicts.
```

To start again the algorithm with another (or, the same) parameters, define `restart` properly to clean the info stored in the algorithm object.

##### Run Monte Carlo experiments

To run a monte-carlo experiment, first pass algorithm class and all data needed by your algorithm to `MAB_MonteCarlo`. The monte-carlo experiments can use multi-processing to boost the speed. (This feature may be changed later for better performance.)

```python
from MAB_algorithm import *
armlist = [heavyTailArm(name=i, maxMomentOrder=2, mean=i/10, mainbound=1.5) for i in range(10)]
mctest = MAB_MonteCarlo(DSEE, arms=armlist, w=10) # `w` is the initial parameter needed by DSEE
gen = mctest.run_monte_carlo_to_list(repeatTimes=1000, iterations=10000, useCores=8)
# Use 8 processes, for 1000 independent expeiment of time step 10000.
i = 0
for ans in gen:
    # `ans` is also a dict containing necessary info like average reward, regret,
    # while information of each experiment is dropped.
    # If every detail is needed, pass `needDetails=True` to `run_monte_carlo_to_list`.
    print(ans)
    i += 1
```

##### MAB with multi-players

> To be implemented

### Contribute

To contribute to this repo (e.g. add benchmark algorithm, arm with other distribution), please create pull request, or create [issue](https://github.com/Antares0982/MAB-algorithm-template/issues) to tell me what you need.
