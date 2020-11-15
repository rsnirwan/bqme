import pytest
import numpy as np
import pystan
from scipy.stats import norm, gamma, lognorm, weibull_min

from bqme.distributions import Normal, Gamma, Lognormal, Weibull

distributions = [
    (Normal(mu=1., sigma=2., name='n'), norm(loc=1., scale=2.)),
    (Gamma(alpha=1., beta=2., name='g'), gamma(a=1., scale=1./2.)),
    (Lognormal(mu=1., sigma=2., name='l'), lognorm(s=2., scale=np.exp(1.))),
    (Weibull(alpha=1., sigma=2., name='w'), weibull_min(c=1., scale=2.)),
]

### test parameter equivalence of scipy and stan
code = lambda dist_name: f"""parameters {{real x;}}
transformed parameters {{real lprob = {dist_name}_lpdf(x| 2., 0.8);}}
model {{x ~ {dist_name}(2.,0.8);}}"""

def test_normal_print():
    mu = Normal(0, 1, name='mu')
    mu2 = Normal(1., 2, name='mu2')
    assert str(mu) == 'Normal(mu=0, sigma=1, name="mu")'
    assert str(mu2) == 'Normal(mu=1.0, sigma=2, name="mu2")'

def test_normal_code():
    code = Normal(0., 1., name='mu').code()
    exp_out = {
            'parameter': 'real mu;', 
            'prior': 'mu ~ normal(0.0, 1.0);'
        }
    assert code == exp_out

@pytest.mark.slow
def test_normal_parameters():
    model_stan = pystan.StanModel(model_code=code('normal'))
    samples = model_stan.sampling()
    s = samples.extract('x')['x']
    s_lp = samples.extract('lprob')['lprob']
    model_scipy = norm(loc=2., scale=0.8)
    s_lp2 = model_scipy.logpdf(s)
    assert np.allclose(s_lp, s_lp2)

def test_gamma_print():
    alpha = Gamma(1, 1, name='alpha')
    beta = Gamma(1, 1.2, name='somethingelse')
    assert str(alpha) == 'Gamma(alpha=1, beta=1, name="alpha")'
    assert str(beta) == 'Gamma(alpha=1, beta=1.2, name="somethingelse")'

def test_gamma_code():
    code = Gamma(1, 1., name='beta').code()
    exp_out = {
            'parameter': 'real<lower=0> beta;', 
            'prior': 'beta ~ gamma(1, 1.0);'
        }
    assert code == exp_out

@pytest.mark.slow
def test_gamma_parameters():
    model_stan = pystan.StanModel(model_code=code('gamma'))
    samples = model_stan.sampling()
    s = samples.extract('x')['x']
    s_lp = samples.extract('lprob')['lprob']
    model_scipy = gamma(a=2., scale=1./0.8)
    s_lp2 = model_scipy.logpdf(s)
    assert np.allclose(s_lp, s_lp2)

def test_lognormal_print():
    mu = Lognormal(1, 1, name='mu')
    sigma = Lognormal(1, 1.2, name='s')
    assert str(mu) == 'Lognormal(mu=1, sigma=1, name="mu")'
    assert str(sigma) == 'Lognormal(mu=1, sigma=1.2, name="s")'
    assert repr(mu) == 'Lognormal(mu=1, sigma=1, name="mu")'
    assert repr(sigma) == 'Lognormal(mu=1, sigma=1.2, name="s")'

def test_lognormal_code():
    code = Lognormal(1, 1., name='sigma').code()
    exp_out = {
            'parameter': 'real<lower=0> sigma;',
            'prior': 'sigma ~ lognormal(1, 1.0);'
        }
    assert code == exp_out

@pytest.mark.slow
def test_lognormal_parameters():
    model_stan = pystan.StanModel(model_code=code('lognormal'))
    samples = model_stan.sampling()
    s = samples.extract('x')['x']
    s_lp = samples.extract('lprob')['lprob']
    model_scipy = lognorm(s=0.8, scale=np.exp(2.))
    s_lp2 = model_scipy.logpdf(s)
    assert np.allclose(s_lp, s_lp2)

def test_weibull_print():
    alpha = Weibull(1., 1., name='alpha')
    sigma = Weibull(1., 1.2, name='s')
    assert str(alpha) == 'Weibull(alpha=1.0, sigma=1.0, name="alpha")'
    assert str(sigma) == 'Weibull(alpha=1.0, sigma=1.2, name="s")'

def test_weibull_code():
    code = Weibull(1., 1., name='sigma').code()
    exp_out = {
            'parameter': 'real<lower=0> sigma;', 
            'prior': 'sigma ~ weibull(1.0, 1.0);'
        }
    assert code == exp_out

@pytest.mark.slow
def test_weibull_parameters():
    model_stan = pystan.StanModel(model_code=code('weibull'))
    samples = model_stan.sampling()
    s = samples.extract('x')['x']
    s_lp = samples.extract('lprob')['lprob']
    model_scipy = weibull_min(c=2., scale=0.8)
    s_lp2 = model_scipy.logpdf(s)
    assert np.allclose(s_lp, s_lp2)

### EXPECTED FAILS

def test_wrong_initialization():
    with pytest.raises(ValueError):
        Normal(0, 0, name='mu')
    with pytest.raises(ValueError):
        Gamma(-1., 2, name='alpha')


### pdf
@pytest.mark.parametrize("bqme_dist, scipy_dist", distributions)
def test_distribution_scipy(bqme_dist, scipy_dist):
    """ test pdf, cdf, logpdf, logcdf, ppf """
    x = np.linspace(1., 3., 10)
    q = np.linspace(0.1, 0.9, 10)
    assert all( bqme_dist.pdf(x) == scipy_dist.pdf(x) )
    assert all( bqme_dist.cdf(x) == scipy_dist.cdf(x) )
    assert all( bqme_dist.logpdf(x) == scipy_dist.logpdf(x) )
    assert all( bqme_dist.logcdf(x) == scipy_dist.logcdf(x) )
    assert all( bqme_dist.ppf(q) == scipy_dist.ppf(q) )

