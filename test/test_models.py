import pytest

from bqme.distributions import Normal, Gamma
from bqme.models import NormalQM, QM
from bqme.settings import BASE_DIR

FILLED_TEMPLATES_PATH = BASE_DIR / 'test' / 'filled_templates'

def test_QM_expected_fail():
    '''input not of type Distribtion'''
    parameters_dict = {
        'a': Normal(0., 1., name='a'),
        'b': 1.
    }
    with pytest.raises(ValueError):
        QM(parameters_dict)


def test_normalQM_print():
    mu = Normal(0., 1., name='mu')
    sigma = Gamma(1., 1., name='sigma')
    model = NormalQM(mu, sigma)
    assert str(model) == 'NormalQM(Normal(mu=0.0, sigma=1.0, name="mu"), Gamma(alpha=1.0, beta=1.0, name="sigma"))'

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

def test_normal_sampling():
    mu = Normal(0., 1., name='mu')
    sigma = Gamma(1., 1.2, name='sigma')
    model = NormalQM(mu, sigma)
    N, q, X = 1000, [0.25, 0.5, 0.75], [-0.1, 0.0, 0.1]
    samples = model.sampling(N, q, X)
    dic = samples.extract(['mu', 'sigma'])
    assert -0.01 < dic['mu'].mean() < 0.01

def test_normal_optimizing():
    mu = Normal(0., 1., name='mu')
    sigma = Gamma(1., 1.2, name='sigma')
    model = NormalQM(mu, sigma)
    N, q, X = 1000, [0.25, 0.5, 0.75], [-0.1, 0.0, 0.1]
    opt = model.optimizing(N, q, X)
    assert -0.01 < opt['mu'].mean() < 0.01


