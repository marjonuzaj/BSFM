## Bayesian Stochastic Frontier Modelling

This repository contains implementations of **Bayesian Stochastic Frontier Models (BSFM)** for efficiency analysis.

The models estimate production frontiers while accounting for inefficiency using different distributions.

## Inefficiency Distributions

The following assumptions for the inefficiency term are supported:

- **Half-Normal Distribution**
- **Exponential Distribution**
- **Lognormal Distribution**

## Package Structure

The `sfm` package is organized as a modular Python package, where each model is implemented in its own file:

```
sfm/
├── __init__.py                  # Exposes the public API
├── normal_hn.py                 # Half-Normal model
├── normal_exponential.py        # Exponential model
└── normal_ln.py                 # Lognormal model
```

Each file contains a single model class, and all models are exposed through the package API for easy access.

### Available models:

- `sfm.HN` → Half-Normal stochastic frontier model
- `sfm.EXP` → Exponential stochastic frontier model
- `sfm.LN` → Lognormal stochastic frontier model

For all methods, the usual production shocks are modelled as $v_{i} \sim N (0,\sigma_{v}^{2})$.

## Example Usage

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sfm

# Create model instance
model1 = sfm.HN(data, 'y')
model2 = sfm.EXP(data, 'y')

# Fit Half-Normal model
model1.fit(nsim=10000, burn=2000)
model2.fit(nsim=10000, burn=2000)

# Print summary
print(model1.summary())
print(model2.summary())

# Plot hist of inef est
plt.hist(model1.inef_est, bins=50, edgecolor='lightgrey', label='HN')
plt.hist(model2.inef_est, bins=50, edgecolor='lightgrey', label='Exp')
plt.xlabel('Inef Est')
plt.xlim(0,max(model.inef_est)+0.1)
plt.legend()
plt.show()
```
