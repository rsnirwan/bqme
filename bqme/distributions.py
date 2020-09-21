from bqme.variables import ContinuousVariable, PositivContinuousVariable

class Distribution:
    """
    Base class for all distribution
    """
    def __init__(self,
            parameters_dict: 'dict used for printing only',
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
