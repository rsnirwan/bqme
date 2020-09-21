import pytest
from bqme.distributions import Normal

def test_print_normal():
    mu = Normal(0, 1, name='mu')
    assert mu.__str__() == 'Normal(mu=0, sigma=1, name="mu")'

def test_wrong_initialization():
    with pytest.raises(ValueError):
        Normal(0, 0, name='mu')
