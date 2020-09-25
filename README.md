# Bayesian Quantile Matching Estimation using Order Statistics


BQME is a package that allows users to fit a distribution to observed quantile data. The package uses Order Statistics as the noise model, which is more robust than e.g. Gaussian noise model (mean squared error). The paper describing the theory can be found on arxiv: [https://arxiv.org/abs/2008.06423](https://arxiv.org/abs/2008.06423). The notebooks for the experiments in the paper are moved to [https://github.com/RSNirwan/BQME_experiments](https://github.com/RSNirwan/BQME_experiments).


## Install

Install latest release via `pip`

```shell
pip install bqme
```

Latest development version can be installes as follows (@todo make develop branch the default).
Clone the repository and install via pip

```shell
git clone https://github.com/RSNirwan/bqme
cd bqme
pip install .
```

Install with dev dependencies 

```shell
git clone https://github.com/RSNirwan/bqme
cd bqme
pip install -e .[dev]
```
if using ZSH, do the following  `pip install -e ".[dev]"`


## Usage

To fit a Normal distribution to observed quantile data, we do the following. Note that the likelihood is not a Normal distribution, but the order statistics of the observed quantiles assuming the underlying distribution is a Normal.

```python
from bqme.distributions import Normal, Gamma
from bqme.models import NormalQM

N, q, X = 100, [0.25, 0.5, 0.75], [-0.1, 0.3, 0.8]

# define priors
mu = Normal(0, 1, name='mu')
sigma = Gamma(1, 1, name='sigma')

# define likelihood
model = NormalQM(mu, sigma)

# sample the posterior
samples = model.sampling(N, q, X)  # returns a stan fit object

# extract samples
mu_samples = samples.extract('mu')['mu']
sigma_samples = samples.extract('sigma')['sigma']
```

We can also look at the generated stan code and optimize the parameters (MAP) instead of sampling the posterior

```python
mu = Normal(0, 1, name='mu')
sigma = Gamma(1, 1, name='sigma')
model = NormalQM(mu, sigma)

# print generated stan code
print(model.code)

# optimize
N, q, X = 100, [0.25, 0.5, 0.75], [-0.1, 0.3, 0.8]
opt = model.optimizing(N, q, X)

# extract optimized parameters
mu_opt = opt['mu']
sigma_opt = opt['sigma']
```

## (so far) Available distributions (prior) and models

distributions/priors (import from `bqme.distributions`): 

* `Normal(mu:float, sigma:float)`
* `Gamma(alpha:float, beta:float)`

models (import from `bqme.models`):

* `NormalQM(mu:distribution, sigma:distribution)`

Inputs to the models need to be distributions.

## Todos

- [ ] make package available on PyPI
- [ ] tag/release on github
- [ ] add code coverage
- [ ] testing with nox
- [ ] use sphinx as documentation tool
- [ ] add Mixture-model
- [ ] implement fit.ppf(q), fit.cdf(x), fit.pdf(x), ...
