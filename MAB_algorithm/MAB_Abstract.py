try:
    from typing import TYPE_CHECKING, List
except ImportError:
    TYPE_CHECKING = False

if TYPE_CHECKING:
    import pandas as pd


class MAB_Runnable(object):
    """Abstract class for MAB runnable classes."""
    __slots__ = []
    def run_to_pdframe(self, number_of_iterations: int) -> "pd.DataFrame":
        raise NotImplementedError()

    def gen_regret_ub_curve(self, number_of_iterations: int) -> List[float]:
        raise NotImplementedError()
