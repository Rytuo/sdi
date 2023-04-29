import gstools as gs
import numpy as np

from sdinterp.interpolator import Interpolator


class Simple(Interpolator):
    """
    Simple kriging.
    Simple kriging is used to interpolate data with a given mean.
    See `gstools.krige.methods.Simple`

    :param cov_model: covariance model, gstools.covmodel.base.CovModel
    :param points: spacial data coordinates of shape (n, 2)
    :param target: spacial data target value of shape (n,)
    """

    def __init__(
            self,
            cov_model: gs.CovModel,
            points: np.ndarray,
            target: np.ndarray,
    ):
        model = gs.krige.Simple(cov_model, points.T, target, exact=False)
        self.model = lambda coords: model(coords.T, post_process=False, store=False)[0]
