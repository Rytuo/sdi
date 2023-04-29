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
        -52.04591504, -51.66686323, -51.2879001, -50.90904836,
        -50.53023543, -50.15139336, -49.77250731, -49.39359834,
        -49.01469616, -48.63581509
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
        -52.04591503, -51.66686322, -51.28790009, -50.90904835,
        -50.53023541, -50.15139335, -49.7725073, -49.39359833,
        -49.01469616, -48.63581508
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
        -52.04591503, -51.66686322, -51.28790009, -50.90904835,
        -50.53023541, -50.15139335, -49.7725073, -49.39359833,
        -49.01469616, -48.63581508
    ])))
