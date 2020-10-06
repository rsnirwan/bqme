from typing import Dict

class FitObject:
    """
    Base class for the fit object
    """
    def __getattr__(self, attr:str) -> 'numpy.ndarray':
        """
        allows to extract model parameters from fit object as attributes
        """
        try:
            ret = self._access_parameter(attr)
        except self._catch_error_access_parameter:
            raise AttributeError( f"Object '{self.__class__.__name__}' has no attribute '{attr}'")
        return ret


class FitObjectSampling(FitObject):
    """
    Fit object using posterior samples of the model.
    This is an extension of the 'StanFit4Model'-type by composition.
    """
    def __init__(self, model:'QM', stan_fit_object:'StanFit4Model'):
        self.model = model
        self.stan_obj = stan_fit_object
        self._catch_error_access_parameter = ValueError

    def _access_parameter(self, attr:str) -> 'numpy.ndarray':
        return self.stan_obj.extract(attr)[attr]


class FitObjectOptimizing(FitObject):
    """
    Fit object using MAP estimate of the model
    """
    def __init__(self, model:'QM', opt_parameters:Dict):
        self.model = model
        self.opt = opt_parameters
        self._catch_error_access_parameter = KeyError

    def _access_parameter(self, attr:str) -> 'numpy.ndarray':
        return self.opt[attr]
