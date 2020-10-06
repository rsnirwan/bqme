import pytest

from bqme.distributions import Normal, Gamma, Lognormal, Weibull
from bqme.models import QM, NormalQM, GammaQM, LognormalQM, WeibullQM


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="run slow tests")

def pytest_runtest_setup(item):
    if 'slow' in item.keywords and not item.config.getvalue("slow"):
        pytest.skip("need --slow option to run")

##### DEFINE FIXTURES HERE

@pytest.fixture(scope='session')
def normal_compiled_model():
    mu = Normal(0., 1., name='mu')
    sigma = Gamma(1., 1.2, name='sigma')
    model = NormalQM(mu, sigma)
    model.compile()
    return  model

@pytest.fixture(scope='module')
def gamma_compiled_model():
    alpha = Gamma(1., 1., name='alpha')
    beta = Gamma(1., 1., name='beta')
    model = GammaQM(alpha, beta)
    model.compile()
    return  model

@pytest.fixture(scope='module')
def lognormal_compiled_model():
    mu = Normal(1., 1., name='mu')
    sigma = Lognormal(1., 1., name='sigma')
    model = LognormalQM(mu, sigma)
    model.compile()
    return  model

@pytest.fixture(scope='module')
def weibull_compiled_model():
    alpha = Weibull(1., 1., name='alpha')
    sigma = Weibull(1., 1., name='sigma')
    model = WeibullQM(alpha, sigma)
    model.compile()
    return  model

