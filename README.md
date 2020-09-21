# Bayesian Quantile Matching Estimation using Order Statistics


BQME is a package that allows users to fit a distribution to observed quantile data. The package uses Order Statistics as the noise model, which is more robust than e.g. Gaussian noise model (mean squared error). The paper describing the theory can be found on arxiv: [https://arxiv.org/abs/2008.06423](https://arxiv.org/abs/2008.06423). The notebooks for the experiments in the paper are moved to [https://github.com/RSNirwan/BQME_experiments](https://github.com/RSNirwan/BQME_experiments).


## Install

Clone the repository and install via pip

```shell
git clone https://github.com/RSNirwan/bqme
cd bqme
pip install .
```


## Usage

To fit a Normal distribution to observed quantile data, we do

```python
from bqme.distributions import Normal, Gamma
from bqme.models import Normal_qm

N, q, X = 100, [0.25, 0.5, 0.75], [-0.1, 0.3, 0.8]

# define prior
mu = Normal(0, 1, name='mu')
sigma = Gamma(0, 1, name='sigma)

# define likelihood
model = Normal_qm(mu, sigma)

# fit model
fit = model.sampling(N, q, X)
```

## Todos

- [ ] make package available on PyPI
- [ ] use sphinx as documentation tool