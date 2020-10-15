# Bayesian Quantile Matching Estimation using Order Statistics


BQME is a package that allows users to fit a distribution to observed quantile data. The package uses Order Statistics as the noise model, which is more robust than e.g. Gaussian noise model (mean squared error). The paper describing the theory can be found on arxiv: [https://arxiv.org/abs/2008.06423](https://arxiv.org/abs/2008.06423). The notebooks for the experiments in the paper are moved to [https://github.com/RSNirwan/BQME_experiments](https://github.com/RSNirwan/BQME_experiments).

BQME generates stan-code that implements the matching and then uses stan's `sampling` and `optimizing` functions for posterior samples and MAP estimate, respectively.


## Install

Install latest release via `pip`

```shell
pip install bqme
```

For latest development version clone the repository and install via pip

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

Here, we fit a Normal distribution to observed quantile data using order statistics of the observed quantiles.
Note that the likelihood is not a Normal distribution, but the order statistics of the observed quantiles assuming the underlying distribution is a Normal.

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
fit = model.sampling(N, q, X)

# extract posterior samples
mu_posterior = fit.mu
sigma_posterior = fit.sigma

# get stan sample object
stan_samples = fit.stan_obj
```

We can also look at the generated stan code and optimize the parameters (MAP) instead of sampling the posterior.

```python
from bqme.distributions import Normal, Gamma
from bqme.models import NormalQM

mu = Normal(0, 1, name='mu')
sigma = Gamma(1, 1, name='sigma')
model = NormalQM(mu, sigma)

# print generated stan code
print(model.code)

# optimize
N, q, X = 100, [0.25, 0.5, 0.75], [-0.1, 0.3, 0.8]
fit = model.optimizing(N, q, X)

# extract optimized parameters
mu_opt = fit.mu
sigma_opt = fit.sigma
```

## Available prior distributions and likelihoods

distributions/priors (import from `bqme.distributions`): 

* [x] `Normal(mu:float, sigma:float, name:str)`
* [x] `Gamma(alpha:float, beta:float, name:str)`
* [x] `Lognormal(mu:float, sigma:float, name:str)`
* [x] `Weibull(alpha:float, sigma:float, name:str)`
* [ ] `InvGamma`
* [ ] `...`


models/likelihoods (import from `bqme.models`):

* [x] `NormalQM(mu:distribution, sigma:distribution)`
* [x] `GammaQM(alpha:distribution, beta:distribution)`
* [x] `LognormalQM(mu:distribution, sigma:distribution)`
* [x] `WeibullQM(alpha:distribution, sigma:distribution)`
* [ ] `InvGammaQM`
* [ ] `...`

Inputs to the models need to be distributions.

## Todos

- [x] make package available on PyPI
- [x] tag/release on github
- [x] github actions for testing on different os and versions
- [x] use sphinx as documentation tool
- [ ] add Mixture-model
- [ ] implement fit.ppf(q), fit.cdf(x), fit.pdf(x), ...
