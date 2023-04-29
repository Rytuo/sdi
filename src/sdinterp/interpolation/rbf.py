import numpy as np
from scipy.interpolate import RBFInterpolator

from sdinterp.interpolator import Interpolator


class Rbf(Interpolator):
    """
    Radial basis function interpolation.
    See `scipy.interpolate._rbfinterp.RBFInterpolator`

    :param points: spacial data coordinates of shape (n, 2)
    :param target: spacial data target value of shape (n, )
    :param neighbors: amount of neighbours processed for each point during evaluation, default is 100
    :param smoothing: smoothing parameter, default is 0
    :param degree: degree used in rbf kernel, default is 1
    :param kernel: type of RBF, should be one of 'linear', 'thin_plate_spline', 'cubic'. Default is 'cubic'
    """

    def __init__(
            self,
            points: np.ndarray,
            target: np.ndarray,
            neighbors: int = 100,
            smoothing: float = 0,
            degree: int = 1,
            kernel: str = 'cubic',
    ):
        if len(points.shape) != 2 or points.shape[1] != 2:
            raise Exception(f"Invalid points dimensions, must be (n, 2), got {points.shape}")
        n = len(points)
        if len(target.shape) != 1 or len(target) != n:
            raise Exception(f"Invalid target dimension, must be ({n},), got {target.shape}")

        if kernel not in ['linear', 'thin_plate_spline', 'cubic']:
            raise Exception(f"Invalid kernel, must be one of 'linear', 'thin_plate_spline', 'cubic', got {kernel}")

        if smoothing == 0:
            # make points unique
            unique_points, inverse_index = np.unique(points, axis=0, return_inverse=True)
            unique_target = [[] for _ in range(len(unique_points))]
            for i, ind in enumerate(inverse_index):
                unique_target[ind].append(target[i])
            unique_target = np.fromiter(map(np.mean, unique_target), dtype=float)
        else:
            unique_points = points
            unique_target = target

        self.model = RBFInterpolator(
            unique_points,
            unique_target,
            kernel=kernel,
            neighbors=min(neighbors, len(target)),
            smoothing=smoothing,
            degree=degree,
        )
