# Spacial data interpolation

This package provides single interface for different interpolation methods.

## Example

Regular usage of ordinary kriging method

```python
import numpy as np
from sdi.variogram import calc_empirical, get_model, fit_model
from sdi.interpolation.kriging import Ordinary

# prepare data
n = 100
points = np.random.random((n, 2))
target = np.random.random((n,))

# get empirical variogram and fit model to it
bin_center, gamma = calc_empirical(points, target)
cov_model = get_model("exponential")
fit_model(cov_model, bin_center, gamma)

# init interpolation model
model = Ordinary(cov_model, points, target)

# evaluate model
prediction = model(np.random.random((1000, 2)))
```

## Supported methods

- Simple Kriging
- Ordinary Kriging
- Universal Kriging
- Inverse Distance Weighting
- Nearest Neighbours
- Linear
- Radial basis function

### Supported covariance models

- Gaussian
- Exponential
- Matern
- Integral
- Stable
- Rational
- Cubic
- Linear
- Circular
- Spherical

## Requirements

- gstools >= 1.4.1
- numpy >= 1.23.5
- scipy >= 1.10.0

## Contact

You can contact me via [popov.aleksandr.v01@gmail.com](popov.aleksandr.v01@gmail.com)
