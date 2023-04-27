from typing import Tuple

import numpy as np

import gstools as gs


def calc_empirical(
        points: np.ndarray,
        target: np.ndarray,
        sampling_size: int = None,
) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Estimates the empirical variogram.
    See `gstools.variogram.variogram`

    :param points: spacial data coordinates of shape (n, 2)
    :param target: spacial data target value of shape (n, )
    :param sampling_size: number of data points to sample randomly, use for large datasets
    :return: (bin_edges, gamma) - points of variogram
    """
    return gs.vario_estimate(points.T, target, sampling_size=sampling_size)
