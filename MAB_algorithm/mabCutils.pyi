from typing import Tuple


class mabarray(object):
    def __init__(self, maxsize: int) -> None: ...
    def add(self, v: float) -> None: ...
    def avg(self) -> float: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> float: ...
    def __setitem__(self, index: int, value: float): ...


def getCatoniMean(
    v: float,
    iteration: int,
    guess: float,
    nd: mabarray,
    tol: float
) -> Tuple[float, int]: ...


def heavytail_dist_pdf(
    alpha: float,
    beta: float,
    coef: float,
    maxmomentorder: float,
    x: float
) -> float: ...
