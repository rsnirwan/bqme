from typing import Dict, Tuple, List

import numpy as np
from scipy.stats import norm, gamma, lognorm, weibull_min

from bqme.variables import ContinuousVariable, PositiveContinuousVariable
from bqme.variables import Variable


class Distribution:
    """
    Base class for all distribution
    Key and values of `parameters_dict` needs the same order as the 
    input to the distribution in stan.
    """
    def __init__(self,
            parameters_dict: Dict[str, Variable],
            name: str):
        self.name = name
        self.parameters_dict = parameters_dict

    def __str__(self):
        params = {name:param.value for name, param in 
                    self.parameters_dict.items()}
        return self.__class__.__name__ + '(' +  \
            ', '.join([f'{k}={v}' for k,v in params.items()]) + \
            f', name="{self.name}"' + ')'

    def __repr__(self):
        return self.__str__()

    def domain(self): pass

    def _stan_code(self) -> Dict[str, str]:
        #real is hard coded

        #parameter initialization
        lower, upper = self.domain()
        parameter = f'real$ {self.name};'
        l = f'lower={lower}' if lower > float('-inf') else ''
        r = f'upper={upper}' if upper < float('inf') else ''
        if l and r:
            parameter = parameter.replace('$', f'<{l}, {r}>')
        elif l or r:
            parameter = parameter.replace('$', f'<{l if l else r}>')
        else:
            parameter = parameter.replace('$', '')

        #prior
        values = ", ".join([
            str(param.value) for param in self.parameters_dict.values()
            ])
        prior = f'{self.name} ~ {self.__class__.__name__.lower()}({values});'

        return {'parameter':parameter, 'prior':prior}


    def code(self) -> Dict[str, str]:
        return self._stan_code()


    def pdf(self, x:List[float]) -> np.ndarray:
        return self._distribution.pdf(x)


    def cdf(self, x:List[float]) -> np.ndarray:
        return self._distribution.cdf(x)


    def logpdf(self, x:List[float]) -> np.ndarray:
        return self._distribution.logpdf(x)

    
    def logcdf(self, x:List[float]) -> np.ndarray:
        return self._distribution.logcdf(x)


    def ppf(self, q:List[float]) -> np.ndarray:
        return self._distribution.ppf(q)
        



class Normal(Distribution):
    """
    Container for Normal Distribution

    Parameters
    ----------
    mu : float
        location of the normal
    sigma : float
        standard deviation of the normal
    name : str

    Examples
    ________
    >>> Normal(0, 1, name='mu')
    Normal(mu=0, sigma=1, name="mu")
    >>> Normal(0, 1, name='mu').code()
    {'parameter': 'real mu;', 'prior': 'mu ~ normal(0, 1);'}
    """
    def __init__(self,
            mu: float,
            sigma: float,
            name: str):
        self.mu = ContinuousVariable(mu, name='mu')
        self.sigma = PositiveContinuousVariable(sigma, name='sigma')
        self.name = name
        self._distribution = norm(loc=self.mu.value, scale=self.sigma.value)
        parameters_dict = {'mu': self.mu, 'sigma': self.sigma}
        super().__init__(parameters_dict, self.name)

    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))



class Gamma(Distribution):
    """
    Container for Gamma Distribution

    Parameters
    ----------
    alpha : float
        shape of the Gamma
    beta : float
        rate of the Gamma
    name : str

    Examples
    ________
    >>> Gamma(1., 1., name='sigma')
    Gamma(alpha=1.0, beta=1.0, name="sigma")
    >>> Gamma(1.1, 1.0, name='sigma').code()
    {'parameter': 'real<lower=0> sigma;', 'prior': 'sigma ~ gamma(1.1, 1.0);'}
    """
    def __init__(self, 
            alpha: float,
            beta: float,
            name: str):
        self.alpha = PositiveContinuousVariable(alpha, name='alpha')
        self.beta = PositiveContinuousVariable(beta, name='beta')
        self.name = name
        self._distribution = gamma(a=self.alpha.value, scale=1./self.beta.value)
        parameters_dict = {'alpha':self.alpha, 'beta':self.beta}
        super().__init__(parameters_dict, self.name)

    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))


class Lognormal(Distribution):
    """
    Container for Lognormal Distribution

    Parameters
    ----------
    mu : float
        log(rv) has loc mu
    sigma : float
        log(rv) has scale sigma
    name : str

    Examples
    ________
    >>> Lognormal(1., 1., name='sigma')
    Lognormal(mu=1.0, sigma=1.0, name="sigma")
    >>> Lognormal(1.1, 1.0, name='sigma').code()
    {'parameter': 'real<lower=0> sigma;', 'prior': 'sigma ~ lognormal(1.1, 1.0);'}
    """
    def __init__(self,
            mu: float,
            sigma: float,
            name: str):
        self.mu = ContinuousVariable(mu, name='mu')
        self.sigma = PositiveContinuousVariable(sigma, name='sigma')
        self.name = name
        # for lognorm parameterization see scipy documentation
        self._distribution = lognorm(s=self.sigma.value, scale=np.exp(self.mu.value))
        parameters_dict = {'mu':self.mu, 'sigma':self.sigma}
        super().__init__(parameters_dict, self.name)

    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))


class Weibull(Distribution):
    """
    Container for Weibull Distribution

    Parameters
    ----------
    alpha : float
        shape of the weibull
    sigma : float
        scale of the weibull - equivalent to 1/rate
    name : str

    Examples
    ________
    >>> Weibull(1., 1., name='sigma')
    Weibull(alpha=1.0, sigma=1.0, name="sigma")
    >>> Weibull(1.1, 1.0, name='sigma').code()
    {'parameter': 'real<lower=0> sigma;', 'prior': 'sigma ~ weibull(1.1, 1.0);'}
    """
    def __init__(self, 
            alpha: float,
            sigma: float,
            name: str):
        self.alpha = PositiveContinuousVariable(alpha, name='alpha')
        self.sigma = PositiveContinuousVariable(sigma, name='sigma')
        self.name = name
        self._distribution = weibull_min(c=self.alpha.value, scale=self.sigma.value)
        parameters_dict = {'alpha':self.alpha, 'sigma':self.sigma}
        super().__init__(parameters_dict, self.name)

    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))
