import logging
from inspect import isfunction

import numpy as np

__all__ = [
    "NewtonIteration"
]


class NewtonIteration(object):
    """This class only contains static functions for Newton Iteration."""

    @staticmethod
    def iter_once(f, df, x, *args, **kwargs) -> float:
        return x-f(x, *args, **kwargs)/df(x, *args, **kwargs)

    @staticmethod
    def iter_full(f, df, x0, tol, maxsteps, *args, **kwargs) -> float:
        """
        A simple newton iteration method.

        Args:
            tol (:param:`Union[float, function]`): Accept float or function. For example,
                if the derivative at zero point is too small, one can change tol to::
                    tol = tol*df(x, *args, **kwargs)
                to acquire a more accurate solution.
        """
        if isfunction(tol):
            tolisfunc = True
            thistol = tol(x0, *args, **kwargs)
        else:
            tolisfunc = False
            thistol = tol
        if thistol <= 0:
            raise ValueError("TOL should be positive")

        x = x0
        a = f(x, *args, **kwargs)
        i = 0
        while np.abs(a) > thistol and i < maxsteps:
            d = df(x, *args, **kwargs)
            x = x-a/d
            a = f(x, *args, **kwargs)
            i += 1
            if tolisfunc:
                thistol = tol(x, *args, **kwargs)
                if thistol <= 0:
                    raise ValueError("TOL function should be positive")

        if i == maxsteps and np.abs(a) > thistol:
            logging.getLogger(__name__).warning(
                f"Newton method: didn't converge after {maxsteps} iterations.")
        return x
