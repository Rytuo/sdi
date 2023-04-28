import numpy as np
from sdi.interpolation.kriging import Simple, Ordinary, Universal

from sdi.variogram import calc_empirical, get_cov_model, fit_cov_model


def test_simple_kriging():
    x, y = np.meshgrid(np.linspace(-100, 100, 10), np.linspace(-100, 100, 10), indexing='ij')
    x, y = x.flatten(), y.flatten()
    points = np.stack((x, y), axis=1)
    target = np.linspace(-100, 0, len(points))

    bin_centers, gamma = calc_empirical(points, target)
    cov_model = get_cov_model("exponential")
    fit_cov_model(cov_model, bin_centers, gamma)

    model = Simple(cov_model, points, target)
    z = model(np.linspace((-5, 5), (0, 30), 10))

    assert np.all(np.isclose(0, z - np.asarray([
        -52.37441146, -46.86769572, -52.09656402, -45.4782057,
        -51.81871121, -44.08878354, -51.5408544, -42.69946934,
        -51.26299493, -41.30993563
    ])))


def test_ordinary_kriging():
    x, y = np.meshgrid(np.linspace(-100, 100, 10), np.linspace(-100, 100, 10), indexing='ij')
    x, y = x.flatten(), y.flatten()
    points = np.stack((x, y), axis=1)
    target = np.linspace(-100, 0, len(points))

    bin_centers, gamma = calc_empirical(points, target)
    cov_model = get_cov_model("exponential")
    fit_cov_model(cov_model, bin_centers, gamma)

    model = Ordinary(cov_model, points, target)
    z = model(np.linspace((-5, 5), (0, 30), 10))

    assert np.all(np.isclose(0, z - np.asarray([
        -52.37441145, -46.8676957, -52.09656401, -45.47820569,
        -51.8187112, -44.08878352, -51.54085439, -42.69946933,
        -51.26299492, -41.30993563
    ])))


def test_universal_kriging():
    x, y = np.meshgrid(np.linspace(-100, 100, 10), np.linspace(-100, 100, 10), indexing='ij')
    x, y = x.flatten(), y.flatten()
    points = np.stack((x, y), axis=1)
    target = np.linspace(-100, 0, len(points))

    bin_centers, gamma = calc_empirical(points, target)
    cov_model = get_cov_model("exponential")
    fit_cov_model(cov_model, bin_centers, gamma)

    model = Universal(cov_model, points, target)
    z = model(np.linspace((-5, 5), (0, 30), 10))

    assert np.all(np.isclose(0, z - np.asarray([
        -52.37441145, -46.8676957, -52.09656401, -45.47820569,
        -51.8187112, -44.08878352, -51.54085439, -42.69946933,
        -51.26299492, -41.30993563
    ])))
