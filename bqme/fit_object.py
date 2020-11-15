from typing import Dict, List

import numpy as np

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

    def _get_samples(self) -> np.ndarray:
        names = self.model.parameters_dict.keys()
        # return shape (#parameters, #samples) - #samples is one for MAP
        return np.array([self._access_parameter(name) for name in names])

    def pdf(self, x:float or List[float], method='mean'):
        """
        Calculates the pdf of x using posterior samples or MAP estimate

        Parameters
        ----------
        x : float or List[float]
            points where the pdf should be evaluated
        method: str, default: 'mean'
            if sampling is used possible values are ('mean', 'median', 'full'). Return values is the mean over all samples if 'mean' is selected. Otherwise median or the full matrix is returned
            if optimizing is used return values are the pdf evaluation of the model at x.

        Returns
        -------
        ret : ndarray
        """
        f = lambda dist, param, x: dist(*param, name='a').pdf(x)
        return self._apply(f, x, method)

    def cdf(self, x:float or List[float], method='mean'):
        """
        Calculates the cdf of x using posterior samples or MAP estimate

        Parameters
        ----------
        x : float or List[float]
            points where the cdf should be evaluated
        method: str, default: 'mean'
            if sampling is used possible values are ('mean', 'median', 'full'). Return values is the mean over all samples if 'mean' is selected. Otherwise median or the full matrix is returned
            if optimizing is used return values are the pdf evaluation of the model at x.

        Returns
        -------
        ret : ndarray
        """
        f = lambda dist, param, x: dist(*param, name='a').cdf(x)
        return self._apply(f, x, method)

    def ppf(self, q:float or List[float], method='full'):
        """
        Calculates the percent point funtion (ppf) of x using posterior samples or MAP estimate

        Parameters
        ----------
        q : float or List[float]
            points where the ppf should be evaluated. Must be in range (0, 1)
        method: str, default: 'mean'
            if sampling is used possible values are ('mean', 'median', 'full'). Return values is the mean over all samples if 'mean' is selected. Otherwise median or the full matrix is returned
            if optimizing is used return values are the pdf evaluation of the model at x.

        Returns
        -------
        ret : ndarray
        """
        f = lambda dist, param, q: dist(*param, name='a').ppf(q)
        return self._apply(f, q, method)


class FitObjectSampling(FitObject):
    """
    Fit object using posterior samples of the model.
    This is an extension of the 'StanFit4Model'-type by composition.
    """
    def __init__(self, model:'QM', stan_fit_object:'StanFit4Model'):
        self.model = model
        self.stan_obj = stan_fit_object
        self._catch_error_access_parameter = ValueError

    def _access_parameter(self, attr:str) -> np.ndarray:
        return self.stan_obj.extract(attr)[attr]

    def _apply(self,
            f:'function',
            x:float or List[float],
            method: str
        ) -> np.ndarray:
        """applys pdf, cdf, ... for each sample to x"""
        posterior_samples = self._get_samples().T
        dist = self.model._distribution
        l = 1 if (type(x)==float or type(x)==int) else len(x)
        ret = np.zeros((posterior_samples.shape[0], l))
        for i, sample in enumerate(posterior_samples):
            ret[i] = f(dist, sample, x)
        if method == 'mean':
            ret = np.mean(ret, axis=0)
        elif method == 'median':
            ret = np.median(ret, axis=0)
        #else return full matrix
        return ret.squeeze()


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

    def _apply(self,
            f:'function',
            x:float or List[float],
            method: str
        ) -> np.ndarray:
        """applys pdf, cdf, ... to x"""
        map_estimate = self._get_samples()
        dist = self.model._distribution
        l = 1 if (type(x)==float or type(x)==int) else len(x)
        return f(dist, map_estimate, x)
