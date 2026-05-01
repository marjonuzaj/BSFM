## Bayesian Stochastic Frontier Modelling

This repository contains implementations of **Bayesian Stochastic Frontier Models (BSFM)** for efficiency analysis.

The models estimate production frontiers while accounting for inefficiency using different distributions.

## Inefficiency Distributions

The following assumptions for the inefficiency term are supported:

- **Half-Normal Distribution**
- **Exponential Distribution**
- **Lognormal Distribution**

## Implementation

The file `models.py` defines the `SFM` class, which contains all methods related to the stochastic frontier model implementation.

Example:

- `fit_halfnormal()` fits the model with $u_{i} \sim N^{+} (0,\sigma_{u}^{2})$
- `fit_exponential()` fits the model with $u_{i} \sim Exp (\lambda)$
- `fit_lognormal()` fits the model with $u_{i} \sim Lognormal (\mu,\sigma_{u}^{2})$.

For all methods, the usual production shocks are modelled as $v_{i} \sim N (0,\sigma_{v}^{2})$.

## Example Usage

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import SFM

# Create model instance
model = SFM(data, 'y')

# Fit Half-Normal model
model.fit_halfnormal(nsim=10000, burn=2000)

# Print summary
print(model.summary())

# Plot hist of inef est
plt.hist(model.inef_est, bins=50, edgecolor='lightgrey')
plt.xlabel('Inef Est')
plt.xlim(0,max(model.inef_est)+0.1)
plt.show()
```
