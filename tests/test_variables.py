import pytest
from bqme.variables import ContinuousVariable, PositiveContinuousVariable

def test_continuousVariable():
    mu = ContinuousVariable(-1., name='mu')
    assert mu.value == -1.
    assert mu.name == 'mu'
    assert mu.upper == float('inf')
    assert mu.lower == float('-inf')

def test_PositiveContinuousVariable():
    sigma = PositiveContinuousVariable(1., name='sigma')
    assert sigma.value == 1.
    assert sigma.name == 'sigma'
    assert sigma.upper == float('inf')
    assert sigma.lower == 0.

def test_PositiveContinuousVariable_expected_fail():
    with pytest.raises(ValueError):
        PositiveContinuousVariable(-1., name='sigma')

