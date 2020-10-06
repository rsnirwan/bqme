import multiprocessing
from typing import Dict, Tuple

from pystan import StanModel

from bqme._settings import STAN_TEMPLATE_PATH
from bqme.distributions import Distribution
from bqme.fit_object import FitObjectSampling, FitObjectOptimizing


multiprocessing.set_start_method("fork") #mac has diffrerent default


class QM:
    """
    Base class for quantile matching (QM) models

    Parameters
    __________
    parameters_dict : Dict
        Keys are internal names for the priors of the model
        e.g. 'mu', 'sigma' for a GaussianQM. Values are the user 
        defined Distributions. Note key must not be identical to value.name.
    """
    def __init__(self, parameters_dict: Dict[str, Distribution]):
        self.parameters_dict = self._check_dict(parameters_dict)
        self.model = None

    def __str__(self):
        return self.__class__.__name__ + '(' +  \
            ', '.join([p.__str__() for p in self.parameters_dict.values()]) + \
            ')'

    def __repr__(self):
        return self.__str__()

    def _check_dict(self, parameters_dict:Dict[str, Distribution]):
        for key, value in parameters_dict.items():
            if not isinstance(value, Distribution):
                raise ValueError(f'Input parameter "{key}" of "{self.__class__.__name__}" needs to be a Distribution (see bqme.distributions), but is of type {type(value)}.')
        return parameters_dict

    def _template_replacements(self) -> Dict['str', 'str']:
        """
        returns a dict that contains keys as template variables
        and values are the strings for the variables.
        Necessary keys: parametersnames, parameters, priors, cdf, lpdf, rng
        """
        distribution_name = self.__class__.__name__.replace("QM", "").lower()
        build = lambda s: '\n    '.join([
                p.code()[s] for p in self.parameters_dict.values()
            ])
        replacements = {
                'parametersnames'   : ', '.join([
                        p.name for p in self.parameters_dict.values()
                    ]),
                'parameters'        : build('parameter'),
                'priors'            : build('prior'),
                'cdf'               : f'{distribution_name}_cdf',
                'lpdf'              : f'{distribution_name}_lpdf',
                'rng'               : f'{distribution_name}_rng',
            }
        return replacements

    def _stan_code(self) -> str:
        with open(STAN_TEMPLATE_PATH) as f:
            code = f.read()
        for k, v in self._template_replacements().items():
            code = code.replace(f'${k}$', v)
        return code

    def _check_domain(self, X):
        minn, maxx = self.domain()
        f = lambda x: not(minn < x < maxx)
        if len(list(filter(f, X))) > 0:
            raise ValueError(f'some elements of X are not in the domain of the model, which is ({minn}, {maxx}).')

    def domain(self): pass

    @property
    def code(self) -> str:
        """ returns the final stan code """
        return self._stan_code()

    def compile(self):
        self.model = StanModel(model_code=self.code)

    def sampling(self, N:int, q:Tuple[float,...], X:Tuple[float,...]) -> 'StanFit4Model':
        self._check_domain(X)
        if self.model is None: self.compile()
        data_dict = {'N':N, 'M':len(q), 'q':q, 'X':X}
        samples = self.model.sampling(data=data_dict)
        return FitObjectSampling(self.model, samples)

    def optimizing(self, N:int, q:Tuple[float,...], X:Tuple[float,...]) -> 'StanFit4Model':
        self._check_domain(X)
        if self.model is None: self.compile()
        data_dict = {'N':N, 'M':len(q), 'q':q, 'X':X}
        opt = self.model.optimizing(data=data_dict)
        return FitObjectOptimizing(self.model, opt)


class NormalQM(QM):
    """
    Quantile matching using Normal distribution

    Parameters
    ----------
    mu : Distribution
        location of the Normal
    sigma : Distribution
        scale of the Normal

    Examples
    --------
    >>> from bqme.distributions import Normal, Gamma
    >>> model = NormalQM(Normal(0., 1., 'mu'), Gamma(1., 1., 'sigma'))
    >>> model
    NormalQM(Normal(mu=0.0, sigma=1.0, name="mu"), Gamma(alpha=1.0, beta=1.0, name="sigma"))
    >>> code = model.code
    """
    def __init__(self, mu:Distribution, sigma:Distribution):
        self.mu = mu
        self.sigma = sigma
        parameters_dict = {'mu': self.mu, 'sigma': self.sigma}
        super().__init__(parameters_dict)

    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))


class GammaQM(QM):
    """
    Quantile matching using Gamma distribution

    Parameters
    ----------
    alpha : Distribution
        Also called the shape of the Gamma
    beta : Distribution
        Also called the rate of the Gamma

    Examples
    --------
    >>> from bqme.distributions import Gamma
    >>> model = GammaQM(Gamma(1., 1. ,'alpha'), Gamma(1., 1., 'beta'))
    >>> model
    GammaQM(Gamma(alpha=1.0, beta=1.0, name="alpha"), Gamma(alpha=1.0, beta=1.0, name="beta"))
    >>> code = model.code
    """
    def __init__(self, alpha:Distribution, beta:Distribution):
        self.alpha = alpha
        self.beta = beta
        parameters_dict = {'alpha': self.alpha, 'beta': self.beta}
        super().__init__(parameters_dict)

    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))


class LognormalQM(QM):
    """
    Quantile matching using Lognormal distribution

    Parameters
    ----------
    mu : Distribution
        location of the corresponding Normal distribution
    sigma : Distribution
        sclae of the corresponding Normal distribution

    Examples
    --------
    >>> from bqme.distributions import Lognormal, Normal
    >>> model = LognormalQM(Normal(1., 1. ,'mu'), Lognormal(1., 1., 'sigma'))
    >>> model
    LognormalQM(Normal(mu=1.0, sigma=1.0, name="mu"), Lognormal(mu=1.0, sigma=1.0, name="sigma"))
    >>> code = model.code
    """
    def __init__(self, mu:Distribution, sigma:Distribution):
        self.mu = mu
        self.sigma = sigma
        parameters_dict = {'mu': self.mu, 'sigma': self.sigma}
        super().__init__(parameters_dict)

    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))


class WeibullQM(QM):
    """
    Quantile matching using Weibull distribution

    Parameters
    ----------
    alpha : Distribution
        Also called the shape of the Weibull
    sigma : Distribution
        Also called the rate of the Weibull

    Examples
    --------
    >>> from bqme.distributions import Weibull
    >>> model = WeibullQM(Weibull(1., 1. ,'alpha'), Weibull(1., 1., 'sigma'))
    >>> model
    WeibullQM(Weibull(alpha=1.0, sigma=1.0, name="alpha"), Weibull(alpha=1.0, sigma=1.0, name="sigma"))
    >>> code = model.code
    """
    def __init__(self, alpha:Distribution, sigma:Distribution):
        self.alpha = alpha
        self.sigma = sigma
        parameters_dict = {'alpha': self.alpha, 'sigma': self.sigma}
        super().__init__(parameters_dict)

    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))
