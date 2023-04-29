import numpy as np
from sdinterp.interpolation import Idw, Linear, Nearest, Rbf


def test_idw():
    x, y = np.meshgrid(np.linspace(-100, 100, 100), np.linspace(-100, 100, 100), indexing='ij')
    x, y = x.flatten(), y.flatten()
    points = np.stack((x, y), axis=1)
    target = np.linspace(-100, 0, len(points))
    model = Idw(points, target)
    z = model(np.linspace((-5, 5), (0, 30), 10))
    assert np.all(np.isclose(0, z - np.asarray([
        -52.47506995, -52.23036518, -51.85396468, -51.47526462,
        -51.31486073, -51.11892329, -50.60488397, -50.39830847,
        -50.28550937, -49.85245487
    ])))


def test_linear():
    x, y = np.meshgrid(np.linspace(-100, 100, 100), np.linspace(-100, 100, 100), indexing='ij')
    x, y = x.flatten(), y.flatten()
    points = np.stack((x, y), axis=1)
    target = np.linspace(-100, 0, len(points))
    model = Linear(points, target)
    z = model(np.linspace((-5, 5), (0, 30), 10))
    assert np.all(np.isclose(0, z - np.asarray([
        -52.45049505, -52.16171617, -51.87293729, -51.58415842,
        -51.29537954, -51.00660066, -50.71782178, -50.4290429,
        -50.14026403, -49.85148515
    ])))


def test_nearest():
    x, y = np.meshgrid(np.linspace(-100, 100, 100), np.linspace(-100, 100, 100), indexing='ij')
    x, y = x.flatten(), y.flatten()
    points = np.stack((x, y), axis=1)
    target = np.linspace(-100, 0, len(points))
    model = Nearest(points, target)
    z = model(np.linspace((-5, 5), (0, 30), 10))
    assert np.all(np.isclose(0, z - np.asarray([
        -52.47524752, -52.46524652, -51.44514451, -51.43514351,
        -51.42514251, -51.40514051, -50.3950395, -50.3750375,
        -50.3650365, -50.3550355
    ])))


def test_rbf():
    x, y = np.meshgrid(np.linspace(-100, 100, 100), np.linspace(-100, 100, 100), indexing='ij')
    x, y = x.flatten(), y.flatten()
    points = np.stack((x, y), axis=1)
    target = np.linspace(-100, 0, len(points))
    model = Rbf(points, target)
    z = model(np.linspace((-5, 5), (0, 30), 10))
    assert np.all(np.isclose(0, z - np.asarray([
        -52.45049505, -52.16171617, -51.87293729, -51.58415842,
        -51.29537954, -51.00660066, -50.71782178, -50.4290429,
        -50.14026403, -49.85148515
    ])))
