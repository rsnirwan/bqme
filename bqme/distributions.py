from typing import Dict, Tuple

from bqme.variables import ContinuousVariable, PositivContinuousVariable
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



class Normal(Distribution):
    """
    Container for Normal Distribution
    """
    def __init__(self,
            mu: float,
            sigma: float,
            name: str):
        self.mu = ContinuousVariable(mu, name='mu')
        self.sigma = PositivContinuousVariable(sigma, name='sigma')
        self.name = name
        parameters_dict = {'mu': self.mu, 'sigma': self.sigma}
        super().__init__(parameters_dict, self.name)

    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))


class Gamma(Distribution):
    """
    Container for Gamma Distribution
    """
    def __init__(self, 
            alpha: float,
            beta: float,
            name: str):
        self.alpha = PositivContinuousVariable(alpha, name='alpha')
        self.beta = PositivContinuousVariable(beta, name='beta')
        self.name = name
        parameters_dict = {'alpha':self.alpha, 'beta':self.beta}
        super().__init__(parameters_dict, self.name)

    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))
