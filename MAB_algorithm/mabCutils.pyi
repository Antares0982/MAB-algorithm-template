from typing import Tuple


class mabnodes(object):
    def __init__(self) -> None: ...
    def add(self, v: float) -> None: ...
    def avg(self) -> float: ...
    def __len__(self) -> int: ...


def getCatoniMean(
    v: float,
    iteration: int,
    guess: float,
    nd: mabnodes,
    tol: float
) -> Tuple[float, int]: ...
