import gstools as gs
import numpy as np

from sdinterp.interpolator import Interpolator


class Universal(Interpolator):
    """
    Universal kriging.
    Universal kriging is used to interpolate given data with a variable mean, that is determined by a functional drift.
    See `gstools.krige.methods.Universal`

    :param cov_model: covariance model, gstools.covmodel.base.CovModel
    :param points: spacial data coordinates of shape (n, 2)
    :param target: spacial data target value of shape (n,)
    :param drift_degree: an integer representing the polynomial order of the drift, default is 0
    """

    def __init__(
            self,
            cov_model: gs.CovModel,
            points: np.ndarray,
            target: np.ndarray,
            drift_degree: int = 0,
    ):
        model = gs.krige.Universal(cov_model, points.T, target, exact=False, drift_functions=drift_degree)
        self.model = lambda coords: model(coords.T, post_process=False, store=False)[0]
