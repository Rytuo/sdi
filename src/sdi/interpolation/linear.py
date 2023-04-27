import numpy as np
from scipy.interpolate import LinearNDInterpolator

from sdi.interpolator import Interpolator


class Linear(Interpolator):
    """
    Linear interpolation.
    See `scipy.interpolate.interpnd.LinearNDInterpolator`

    :param points: spacial data coordinates of shape (n, 2)
    :param target: spacial data target value of shape (n,)
    """

    def __init__(
            self,
            points: np.ndarray,
            target: np.ndarray,
    ):
        self.model = LinearNDInterpolator(points, target)
