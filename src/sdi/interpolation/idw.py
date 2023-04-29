import numpy as np
from scipy.spatial import KDTree

from sdi.interpolator import Interpolator


class Idw(Interpolator):
    """
    Inverse distance weighting with modified Shepard's method.
    See https://en.wikipedia.org/wiki/Inverse_distance_weighting

    :param points: spacial data coordinates of shape (n, 2)
    :param target: spacial data target value of shape (n, )
    :param p: minkowski norm used in calculations, default is 2
    :param k: amount of neighbours processed for each point during evaluation, default is 10
    :param r: maximum radius of neighbours search, default is 1000
    """

    def __init__(
            self,
            points: np.ndarray,
            target: np.ndarray,
            p=2,
            k=10,
            r=1000,
    ):
        sparse_tree = KDTree(points)
        n = len(points)

        def model(dense_points: np.ndarray[float]) -> np.ndarray:
            def shepard(dists: np.ndarray[float], ind: np.ndarray):
                invalid = np.nonzero(ind == n)
                if len(invalid) > 0:
                    ind = np.delete(ind, invalid)
                    dists = np.delete(dists, invalid)

                if len(ind) == 0:
                    return None

                is_zero = np.where(np.isclose(dists, 0))[0]
                if len(is_zero):  # any dist is zero
                    return target[ind[is_zero[0]]]

                w = ((r - dists) / (r * dists)) ** 2
                return (w @ target[ind]) / sum(w)

            all_dists, sparse_indices = sparse_tree.query(dense_points, p=p, k=k, distance_upper_bound=r)

            dense_z = np.empty(len(dense_points))
            for i, (d, sparse_i) in enumerate(zip(all_dists, sparse_indices)):
                dense_z[i] = shepard(d, sparse_i)

            return dense_z

        self.model = model
