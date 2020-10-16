import pytest
import numpy as np

from bqme.fit_object import FitObjectSampling, FitObjectOptimizing

@pytest.mark.slow
def test_fitObjectSampling(normal_compiled_model):
    N, q, X = 100, [0.25, 0.5, 0.75], [-0.1, 0.3, 0.8]
    fit = normal_compiled_model.sampling(N, q, X)
    samples = fit.stan_obj
    assert all(fit.mu == samples.extract('mu')['mu'])
    assert all(fit.sigma == samples.extract('sigma')['sigma'])

@pytest.mark.slow
def test_fitObjectOptimizing(normal_compiled_model):
    N, q, X = 100, [0.25, 0.5, 0.75], [-0.1, 0.3, 0.8]
    fit = normal_compiled_model.optimizing(N, q, X)
    opt = fit.opt
    assert fit.mu == opt['mu']
    assert fit.sigma == opt['sigma']

@pytest.mark.slow
def test_fitObjectSampling_expected_fail(normal_compiled_model):
    N, q, X = 100, [0.25, 0.5, 0.75], [-0.1, 0.3, 0.8]
    fit = normal_compiled_model.sampling(N, q, X)
    # fits parameters are mu and sigma only
    with pytest.raises(AttributeError):
        fit.bla  # bla is not an attribute of fit

@pytest.mark.slow
def test_NormalQM_sampling(normal_compiled_model):
    N, q, X = 100, [0.25, 0.5, 0.75], [0.1, 0.3, 0.8]
    fit = normal_compiled_model.sampling(N, q, X)
    assert fit.pdf(3.) < 0.1
    assert fit.cdf(3.) > 0.9
    assert all(fit.ppf(.9) > 0.)
    assert fit.pdf(3., method='median') < 0.1
    assert fit.cdf(3., method='median') > 0.9
    assert np.all(fit.ppf(.9, method='median') > 0.)

@pytest.mark.slow
def test_NormalQM_optimizing(normal_compiled_model):
    N, q, X = 100, [0.25, 0.5, 0.75], [0.1, 0.3, 0.8]
    fit = normal_compiled_model.optimizing(N, q, X)
    assert fit.pdf(3.) < 0.1
    assert fit.cdf(3.) > 0.9
    assert fit.ppf(.9) > 0.

