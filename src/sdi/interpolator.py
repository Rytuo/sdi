from typing import Callable

import numpy as np


class Interpolator:
    """ Base class for all interpolation methods """

    model: Callable[[list[tuple[float, float]]], np.ndarray[float]] = None

    def __call__(self, points: list[tuple[float, float]]) -> np.ndarray[float]:
        """
        Evaluate method at given points
        :param points: coordinates of points
        :return: interpolated values in target dimension
        """
        return self.model(points)
