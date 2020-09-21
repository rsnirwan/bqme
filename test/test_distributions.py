import pytest
from bqme.distributions import Normal, Gamma

def test_normal_print():
    mu = Normal(0, 1, name='mu')
    mu2 = Normal(1., 2, name='mu2')
    assert mu.__str__() == 'Normal(mu=0, sigma=1, name="mu")'
    assert mu2.__str__() == 'Normal(mu=1.0, sigma=2, name="mu2")'

def test_normal_code():
    code = Normal(0., 1., name='mu').code()
    exp_out = {
            'parameter': 'real mu;', 
            'prior': 'mu ~ normal(0.0, 1.0);'
        }
    assert code == exp_out

def test_gamma_print():
    alpha = Gamma(1, 1, name='alpha')
    beta = Gamma(1, 1.2, name='somethingelse')
    assert alpha.__str__() == 'Gamma(alpha=1, beta=1, name="alpha")'
    assert beta.__str__() == 'Gamma(alpha=1, beta=1.2, name="somethingelse")'

def test_gamma_code():
    code = Gamma(1, 1., name='beta').code()
    exp_out = {
            'parameter': 'real<lower=0> beta;', 
            'prior': 'beta ~ gamma(1, 1.0);'
        }
    assert code == exp_out


def test_wrong_initialization():
    with pytest.raises(ValueError):
        Normal(0, 0, name='mu')
    with pytest.raises(ValueError):
        Gamma(-1., 2, name='alpha')
