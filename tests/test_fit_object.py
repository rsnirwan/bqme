import pytest

from bqme.fit_object import FitObjectSampling, FitObjectOptimizing

@pytest.mark.slow
def test_fitObjectSampling(normal_compiled_model):
    N, q, X = 100, [0.25, 0.5, 0.75], [-0.1, 0.3, 0.8]
    fit = normal_compiled_model.sampling(N, q, X)
    samples = fit.stan_obj
    assert any(fit.mu == samples.extract('mu')['mu'])
    assert any(fit.sigma == samples.extract('sigma')['sigma'])

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

