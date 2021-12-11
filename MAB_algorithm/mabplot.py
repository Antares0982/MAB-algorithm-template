from typing import List, Optional, Union, overload

import matplotlib.pyplot as plt
import pandas as pd

try:
    from typing import TYPE_CHECKING
except ImportError:
    TYPE_CHECKING = False

if TYPE_CHECKING:
    from MAB_algorithm.MAB import MABAlgorithm

__all__ = [
    "plotResult", "singlePlot"
]


@overload
def plotResult(
    algorithms: List['MABAlgorithm'],
    number_of_iterations: int,
    algorithm_names: Optional[List[str]] = None,
    ignore_columns: List[str] = []
):
    ...


def _plot_algorithm(
    algorithms: List['MABAlgorithm'],
    number_of_iterations: int,
    algorithm_names: Optional[List[str]] = None,
    ignore_columns: List[str] = []
):
    alldata = list(map(lambda x: x.run_to_pdframe(
        number_of_iterations), algorithms))
    if algorithm_names is None:
        algorithm_names = list(map(lambda x: x.__class__.__name__, algorithms))
    _plot_dataframe(alldata, algorithm_names, ignore_columns)


@overload
def plotResult(
    alldata: List[List[List[Union[float, int]]]],
    columnNames: List[str],
    algorithm_names: List[str],
    ignore_columns: List[str] = []
):
    ...


def _plot_list(
    alldata: List[List[List[Union[float, int]]]],
    columnNames: List[str],
    algorithm_names: List[str],
    ignore_columns: List[str] = []
):
    try:
        int(alldata[0][0][0])
    except Exception:
        raise ValueError("invalid argument type")
    alldata = list(map(lambda x: pd.DataFrame(
        x, columns=columnNames), alldata))
    _plot_dataframe(alldata, algorithm_names, ignore_columns)


@overload
def plotResult(
    alldata: List[pd.DataFrame],
    algorithm_names: List[str],
    ignore_columns: List[str] = []
):
    ...


def _plot_dataframe(
    alldata: List[pd.DataFrame],
    algorithm_names: List[str],
    ignore_columns: List[str] = []
):
    if type(alldata[0]) is not pd.DataFrame:
        raise TypeError("invalid argument type")
    if len(algorithm_names) != len(alldata):
        raise ValueError("length of argument doesn't match")

    columnNames = alldata[0].columns
    ignore_columns += ["chosen_arm", "iteration"]
    for col in columnNames:
        if col in ignore_columns:
            continue
        for i, name in enumerate(algorithm_names):
            plt.plot(alldata[i].get(col), label=name)
        plt.legend(loc='best')
        plt.title(f"The {col} curve of algorithms")
        plt.show()


def plotResult(*args, **kwargs):
    tp = 0  # 1 is algorithm, 2 is list, 3 is dataframe
    for k in kwargs.keys():
        if k in ["algorithms", "number_of_iterations"]:
            tp = 1
            break
        elif k in ["columnNames"]:
            tp = 2
            break

    if tp == 0:
        try:
            int(args[0][0][0][0])
        except Exception:
            ...
        else:
            tp = 2

    if tp == 0:
        try:
            firstval = args[0][0]
        except Exception:
            raise ValueError("invalid argument")
        if type(firstval) is pd.DataFrame:
            tp = 3
        else:
            tp = 1

    if tp == 1:
        _plot_algorithm(*args, **kwargs)
    elif tp == 2:
        _plot_list(*args, **kwargs)
    elif tp == 3:
        _plot_dataframe(*args, **kwargs)


@overload
def singlePlot(
    algorithm: 'MABAlgorithm',
    number_of_iterations: int,
    algorithm_name: Optional[str] = None,
    use_regret_ub_curve: bool = True,
    ignore_columns: List[str] = []
):
    ...


def _plot_single_algorithm(
    algorithm: 'MABAlgorithm',
    number_of_iterations: int,
    algorithm_name: Optional[str] = None,
    use_regret_ub_curve: bool = True,
    ignore_columns: List[str] = []
):
    curve = None
    pdfr = algorithm.run_to_pdframe(number_of_iterations)
    if algorithm_name is None:
        algorithm_name = algorithm.__class__.__name__
    if use_regret_ub_curve:
        curve = algorithm.gen_regret_ub_curve(number_of_iterations)
    _plot_single_dataframe(pdfr, algorithm_name, curve, ignore_columns)


@overload
def singlePlot(
    data: List[List[Union[float, int]]],
    columnNames: List[str],
    algorithm_name: str,
    regret_ub_curve: Optional[List[float]] = None,
    ignore_columns: List[str] = []
):
    ...


def _plot_single_list(
    data: List[List[Union[float, int]]],
    columnNames: List[str],
    algorithm_name: str,
    regret_ub_curve: Optional[List[float]] = None,
    ignore_columns: List[str] = []
):
    data = pd.DataFrame(data, columns=columnNames)
    _plot_single_dataframe(data, algorithm_name,
                           regret_ub_curve, ignore_columns)


@overload
def singlePlot(
    data: pd.DataFrame,
    algorithm_name: str,
    regret_ub_curve: Optional[List[float]] = None,
    ignore_columns: List[str] = []
):
    ...


def _plot_single_dataframe(
    data: pd.DataFrame,
    algorithm_name: str,
    regret_ub_curve: Optional[List[float]] = None,
    ignore_columns: List[str] = []
):
    ignore_columns += ["chosen_arm", "iteration"]
    for col in data.columns:
        if col in ignore_columns:
            continue
        if regret_ub_curve is not None and col == "regret":
            plt.plot(regret_ub_curve, label="regret upper bound")
        plt.plot(data.get(col), label=algorithm_name)
        plt.legend(loc='best')
        plt.title(f"The {col} curve of algorithm {algorithm_name}")
        plt.show()


def singlePlot(*args, **kwargs):
    tp = 0  # 1 is algorithm, 2 is list, 3 is dataframe
    for k in kwargs.keys():
        if k in ["algorithm", "number_of_iterations", "use_regret_ub_curve"]:
            tp = 1
            break
        elif k in ["columnNames"]:
            tp = 2
            break

    if tp == 0:
        try:
            int(args[0][0][0])
        except Exception:
            ...
        else:
            tp = 2

    if tp == 0:
        firstval = args[0]
        if type(firstval) is pd.DataFrame:
            tp = 3
        else:
            tp = 1

    if tp == 1:
        _plot_single_algorithm(*args, **kwargs)
    elif tp == 2:
        _plot_single_list(*args, **kwargs)
    elif tp == 3:
        _plot_single_dataframe(*args, **kwargs)
