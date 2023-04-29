import numpy as np
from scipy.interpolate import NearestNDInterpolator

from sdinterp.interpolator import Interpolator


class Nearest(Interpolator):
    """
    Nearest-neighbor interpolation.
    See `scipy.interpolate._ndgriddata.NearestNDInterpolator`

    :param points: spacial data coordinates of shape (n, 2)
    :param target: spacial data target value of shape (n,)
    """

    def __init__(
            self,
            points: np.ndarray,
            target: np.ndarray,
    ):
        self.model = NearestNDInterpolator(points, target)
