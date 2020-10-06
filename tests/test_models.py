import pytest

from bqme.distributions import Normal, Gamma, Lognormal, Weibull
from bqme.models import QM, NormalQM, GammaQM, LognormalQM, WeibullQM
from bqme._settings import BASE_DIR

FILLED_TEMPLATES_PATH = BASE_DIR / 'tests' / 'filled_templates'

def test_QM_expected_fail():
    '''input not of type Distribtion'''
    parameters_dict = {
        'a': Normal(0., 1., name='a'),
        'b': 1.   # input should be a Distribution
    }
    with pytest.raises(ValueError):
        QM(parameters_dict)


### NormalQM tests

def test_normalQM_print():
    mu = Normal(0., 1., name='mu')
    sigma = Gamma(1., 1., name='sigma')
    model = NormalQM(mu, sigma)
    assert str(model) == 'NormalQM(Normal(mu=0.0, sigma=1.0, name="mu"), Gamma(alpha=1.0, beta=1.0, name="sigma"))'
    assert repr(model) == 'NormalQM(Normal(mu=0.0, sigma=1.0, name="mu"), Gamma(alpha=1.0, beta=1.0, name="sigma"))'

def test_normalQM_template_replacements():
    mu = Normal(0., 1., name='loc')
    sigma = Gamma(1., 1., name='scale')
    model = NormalQM(mu, sigma)
    replacements = model._template_replacements()
    assert replacements['parametersnames'] == 'loc, scale'
    assert replacements['parameters'] == 'real loc;\n    real<lower=0> scale;'
    assert replacements['priors'] == \
            'loc ~ normal(0.0, 1.0);\n    scale ~ gamma(1.0, 1.0);'
    assert replacements['cdf'] == 'normal_cdf'
    assert replacements['lpdf'] == 'normal_lpdf'
    assert replacements['rng'] == 'normal_rng'

def test_normal_code():
    mu = Normal(0., 1., name='mu')
    sigma = Gamma(1., 1.2, name='sigma')
    model = NormalQM(mu, sigma)
    code = model.code
    with open(FILLED_TEMPLATES_PATH / 'os_normal.stan') as f:
        code_hard_coded = f.read()
    assert code == code_hard_coded

@pytest.mark.slow
def test_normal_sampling(normal_compiled_model):
    N, q, X = 1000, [0.25, 0.5, 0.75], [-0.1, 0.0, 0.1]
    samples = normal_compiled_model.sampling(N, q, X).stan_obj
    dic = samples.extract(['mu', 'sigma'])
    assert -0.01 < dic['mu'].mean() < 0.01

@pytest.mark.slow
def test_normal_optimizing(normal_compiled_model):
    N, q, X = 1000, [0.25, 0.5, 0.75], [-0.1, 0.0, 0.1]
    opt = normal_compiled_model.optimizing(N, q, X)
    assert -0.01 < opt.mu < 0.01


### GammaQM tests

def test_gamma_code():
    alpha = Gamma(1.0, 1.2, name='alpha')
    beta = Gamma(2.1, 2.2, name='beta')
    model = GammaQM(alpha, beta)
    code = model.code
    with open(FILLED_TEMPLATES_PATH / 'os_gamma.stan') as f:
        code_hard_coded = f.read()
    assert code == code_hard_coded

def test_gamma_check_domain_expected_fail():
    alpha = Gamma(1.0, 1.2, name='alpha')
    beta = Gamma(2.1, 2.2, name='beta')
    model = GammaQM(alpha, beta)
    N, q, X = 1000, [0.25, 0.5, 0.75], [-0.1, 1.0, 1.4] #-0.1 is invalid
    with pytest.raises(ValueError):
        model.sampling(N, q, X)
    with pytest.raises(ValueError):
        model.optimizing(N, q, X)

@pytest.mark.slow
def test_gamma_sampling(gamma_compiled_model):
    N, q, X = 1000, [0.25, 0.5, 0.75], [0.1, 1.0, 1.4]
    samples = gamma_compiled_model.sampling(N, q, X).stan_obj
    dic = samples.extract(['alpha', 'beta'])
    assert dic['alpha'].mean() > 0.

@pytest.mark.slow
def test_gamma_optimizing(gamma_compiled_model):
    N, q, X = 1000, [0.25, 0.5, 0.75], [0.1, 1.0, 1.4]
    opt = gamma_compiled_model.optimizing(N, q, X)
    assert opt.alpha > 0.

### LognormalQM tests

def test_lognormal_code():
    mu = Normal(1.0, 1.2, name='mu')
    sigma = Lognormal(2.1, 2.2, name='sigma')
    model = LognormalQM(mu, sigma)
    code = model.code
    with open(FILLED_TEMPLATES_PATH / 'os_lognormal.stan') as f:
        code_hard_coded = f.read()
    assert code == code_hard_coded

def test_lognormal_check_domain_expected_fail():
    mu = Normal(1.0, 1.2, name='mu')
    sigma = Lognormal(2.1, 2.2, name='sigma')
    model = LognormalQM(mu, sigma)
    N, q, X = 1000, [0.25, 0.5, 0.75], [-0.1, 1.0, 1.4] #-0.1 is invalid
    with pytest.raises(ValueError):
        model.sampling(N, q, X)
    with pytest.raises(ValueError):
        model.optimizing(N, q, X)

@pytest.mark.slow
def test_lognormal_sampling(lognormal_compiled_model):
    N, q, X = 1000, [0.25, 0.5, 0.75], [0.1, 1.0, 1.4]
    samples = lognormal_compiled_model.sampling(N, q, X).stan_obj
    dic = samples.extract(['mu', 'sigma'])
    assert dic['mu'].mean() > -1.

@pytest.mark.slow
def test_lognormal_optimizing(lognormal_compiled_model):
    N, q, X = 1000, [0.25, 0.5, 0.75], [0.1, 1.0, 1.4]
    opt = lognormal_compiled_model.optimizing(N, q, X)
    assert opt.mu > -1.

### WeibullQM tests

def test_weibull_code():
    alpha = Gamma(1.0, 1.2, name='alpha')
    sigma = Weibull(2.1, 2.2, name='sigma')
    model = WeibullQM(alpha, sigma)
    code = model.code
    with open(FILLED_TEMPLATES_PATH / 'os_weibull.stan') as f:
        code_hard_coded = f.read()
    assert code == code_hard_coded

def test_weibull_check_domain_expected_fail():
    alpha = Weibull(1.0, 1.2, name='alpha')
    sigma = Weibull(2.1, 2.2, name='sigma')
    model = WeibullQM(alpha, sigma)
    N, q, X = 1000, [0.25, 0.5, 0.75], [-0.1, 1.0, 1.4] #-0.1 is invalid
    with pytest.raises(ValueError):
        model.sampling(N, q, X)
    with pytest.raises(ValueError):
        model.optimizing(N, q, X)

@pytest.mark.slow
def test_weibull_sampling(weibull_compiled_model):
    N, q, X = 1000, [0.25, 0.5, 0.75], [0.1, 1.0, 1.4]
    samples = weibull_compiled_model.sampling(N, q, X).stan_obj
    dic = samples.extract(['alpha', 'sigma'])
    assert dic['alpha'].mean() > 0.

@pytest.mark.slow
def test_weibull_optimizing(weibull_compiled_model):
    N, q, X = 1000, [0.25, 0.5, 0.75], [0.1, 1.0, 1.4]
    opt = weibull_compiled_model.optimizing(N, q, X)
    assert opt.alpha > 0.
