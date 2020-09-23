import multiprocessing
from typing import Dict, Tuple

from pystan import StanModel

from bqme.distributions import Distribution
from bqme.settings import STAN_TEMPLATE_PATH


multiprocessing.set_start_method("fork") #mac has diffrerent default


class QM:
    """
    Base class for quantile matching (QM) models

    Parameters
    _________
    parameters_dict : Dict
        Keys are internal names for the priors of the model
        e.g. 'mu', 'sigma' for a GaussianQM. Values are the user 
        defined Distributions. Note key must not be identical to value.name.
    """
    def __init__(self, parameters_dict: Dict[str, Distribution]):
        self.parameters_dict = parameters_dict
        self.model = None

    def __str__(self):
        return self.__class__.__name__ + '(' +  \
            ', '.join([p.__str__() for p in self.parameters_dict.values()]) + \
            ')'

    def __repr__(self):
        return self.__str__()

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

    def domain(self): pass

    def code(self) -> str:
        """ returns the final stan code """
        return self._stan_code()

    def compile(self):
        self.model = StanModel(model_code=self.code())

    def sampling(self, N:int, q:Tuple[float,...], X:Tuple[float,...]) -> 'StanFit4Model':
        if self.model is None:
            self.compile()
        data_dict = {'N':N, 'M':len(q), 'q':q, 'X':X}
        return self.model.sampling(data=data_dict)


class NormalQM(QM):
    """
    Quantile matching using Normal distribution
    """
    def __init__(self, mu:Distribution, sigma:Distribution):
        self.mu = mu
        self.sigma = sigma
        parameters_dict = {'mu': self.mu, 'sigma': self.sigma}
        super().__init__(parameters_dict)

    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))

