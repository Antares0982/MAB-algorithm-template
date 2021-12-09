import logging
from inspect import isfunction
from typing import Optional

import numpy as np

__all__ = [
    "NewtonIteration",
    "Node",
    "MAB_Nodes"
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


class Node(object):
    __slots__ = [
        "num",
        "next"
    ]

    def __init__(self, num: float) -> None:
        """Store a num and the next node."""
        self.num = num
        self.next: Optional[Node] = None


class MAB_Nodes(object):
    """Represent a list of nodes."""

    def __init__(self) -> None:
        """Store head and tail, and an int `__len` representing length."""
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None
        self.__len = 0

    def add(self, num: float):
        if self.head is None:
            self.head = Node(num)
            self.tail = self.head
        else:
            self.tail.next = Node(num)
            self.tail = self.tail.next
        self.__len += 1

    def __iter__(self):
        """The iterator yields all the floats stored in the list of nodes."""
        p = self.head
        while p is not None:
            yield p.num
            p = p.next

    def avg(self) -> float:
        ss = 0
        for n in self:
            ss += n
        return ss/self.__len

    def __len__(self):
        """Return length of list."""
        return self.__len

    def __repr__(self) -> str:
        """Represents like a list object. Calling `print(nodes)` is not suggested."""
        return str([x for x in self])
