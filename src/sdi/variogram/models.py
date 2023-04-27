import gstools as gs
import numpy as np

MODELS = {
    "gaussian": gs.Gaussian,
    "exponential": gs.Exponential,
    "matern": gs.Matern,
    "integral": gs.Integral,
    "stable": gs.Stable,
    "rational": gs.Rational,
    "cubic": gs.Cubic,
    "linear": gs.Linear,
    "circular": gs.Circular,
    "spherical": gs.Spherical,
}


def get_model(
        model_name: str,
        var: float = 1.0,
        len_scale: float = 1.0,
        nugget: float = 0.0,
) -> gs.CovModel:
    """
    Get model with given parameters.
    See `gstools.covmodel.base`

    :param model_name: one of 'gaussian', 'exponential', 'matern', 'integral', 'stable', 'rational', 'cubic',
    'linear', 'circular', 'spherical'
    :param var: variance, default is 1.0
    :param len_scale: range, default is 1.0
    :param nugget: nugget, default is 0.0
    :return:
    """
    return MODELS[str.lower(model_name)](
        dim=2,
        var=var,
        len_scale=len_scale,
        nugget=nugget,
    )


def fit_model(
        model: gs.CovModel,
        bin_center: np.ndarray,
        gamma: np.ndarray,
        **para_select,
) -> gs.CovModel:
    """
    Fitting the variogram-model to an empirical variogram.
    See `gstools.covmodel.base.CovModel`

    :param model: covariance model
    :param bin_center: bin centers of empirical variogram
    :param gamma: values of empirical variogram
    :param para_select: dictionary with mapping `parameter_name -> isExcluded`, use to deselect parameters from fitting
    :return:
    """
    if ('var' in para_select and para_select['var'] is False) \
            and ('len_scale' in para_select and para_select['len_scale'] is False) \
            and ('param' in para_select and para_select['param'] is False):
        return model
    model.fit_variogram(bin_center, gamma, **para_select)
    return model
