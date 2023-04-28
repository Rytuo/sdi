import numpy as np
import pytest

from sdi.variogram import calc_empirical, get_cov_model, fit_cov_model


def test_calc_empirical():
    points = np.linspace((-100, -100), (100, 100), 100)
    target = np.linspace(-100, 100, 100)
    bin_edges, gamma = calc_empirical(points, target)
    assert np.all(np.isclose(0, bin_edges - np.asarray([
        3.14269681, 9.42809042, 15.71348403, 21.99887764, 28.28427125,
        34.56966486, 40.85505847, 47.14045208, 53.42584569, 59.7112393,
        65.99663291, 72.28202652, 78.56742013, 84.85281374, 91.13820735,
    ])))
    assert np.all(np.isclose(0, gamma - np.asarray([
        5.08598263, 25.47059542, 62.17916431, 115.21163036, 193.93441217,
        301.44707261, 429.37497129, 555.86409061, 698.67667191, 884.94371323,
        1100.55717239, 1327.06633825, 1543.33301669, 1775.92241637, 2041.94129851,
    ])))


@pytest.mark.filterwarnings("ignore::gstools.covmodel.tools.AttributeWarning")
def test_get_model():
    model_names = ["gaussian", "exponential", "matern", "integral", "stable",
                   "rational", "cubic", "linear", "circular", "spherical"]
    for model_name in model_names:
        var, len_scale, nugget = 853, 1531, 351561
        cov_model = get_cov_model(model_name, var, len_scale, nugget)
        assert cov_model.var == var
        assert cov_model.len_scale == len_scale
        assert cov_model.nugget == nugget


def test_fit_model():
    cov_model = get_cov_model("exponential")
    bin_centers = np.linspace(0, 100, 100)
    gamma = np.log(np.linspace(1, 100, 100))
    fit_cov_model(
        cov_model,
        bin_centers,
        gamma
    )
    assert np.isclose(cov_model.var, 3.4814731767969587)
    assert np.isclose(cov_model.len_scale, 24.59081358961471)
    assert np.isclose(cov_model.nugget, 1.0097876141471076)
