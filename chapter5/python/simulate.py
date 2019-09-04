import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from scipy.stats import gamma
from statsmodels.distributions.empirical_distribution import ECDF
from utility import timer

#####################################
# Random variable class definitions #
#####################################

class RVContinuous(object):
    """
    This is a class for defining generic continuous random variables.
    """

    def __init__(self, name = 'unknown', support = (0.0, 1.0), cdf = None, pdf = None, **params):
        """
        In case the random variable has a well-known distribution, providing the name of the random variable and
        **params = parameters of the distribution will set all other arguments automatically.
        Currently a known name can be anything in the list ['gamma']. Dafault is 'unknown'.
        support = support of the pdf, default = (0.0, 1.0)
        params = dict of keyword arguments that are passed to pdf, cdf and inv_cdf
        Either pdf or cdf is required for mean and variance computation. One of them can be omitted.
        """
        # set support and pdf/cdf for known distributions
        if name == 'gamma':
            support = (0.0, np.inf)
            cdf = lambda x, shape, scale: gamma.cdf(x/scale, shape)/scale
            pdf = lambda x, shape, scale: gamma.pdf(x/scale, shape)/scale

        # assign basic attributes
        self.name = name # name of the random variable
        self.a, self.b = support # left and right endpoints of support interval of pdf
        self.params = params # parameters of pdf and cdf
        if pdf is not None:
            self.pdf_ = pdf # family of pdfs without parmeters specified
            self.pdf = lambda x: self.pdf_(x, **self.params) # pdf with parameters specified
        if cdf is not None:
            self.cdf_ = cdf # family of cdfs without parmeters specified
            self.cdf = lambda x: self.cdf_(x, **self.params) # cdf with parameters specified


    def set_params(self, **new_params):
        """
        Resets parameters of the distribution to new_params.
        """
        self.params = new_params
        if hasattr(self, 'pdf'):
            self.pdf = lambda x: self.pdf_(x, **self.params)
        if hasattr(self, 'cdf'):
            self.cdf = lambda x: self.cdf_(x, **self.params)


    def compute_mean(self):
        """
        Computes, sets and returns self.mean = expected value of the random variable.
        """
        # compute mean according to availability of pdf or cdf
        if hasattr(self, 'pdf'):
            # compute mean using pdf
            self.mean = integrate.quad(lambda x: x*self.pdf(x), self.a, self.b)[0]
        elif hasattr(self, 'cdf'):
            # parts of integrand
            plus = lambda x: 1.0 - self.cdf(x)
            minus = lambda x: -self.cdf(x)

            # left limit of integration
            left_lim = self.a

            # decide integrand and correction term according to self.a and self.b being finite/infinite
            if not np.isinf(self.a):
                integrand = plus
                correction = self.a
            elif not np.isinf(self.b):
                integrand = minus
                correction = self.b
            else:
                integrand = lambda x: plus(x) + minus(-x)
                correction = 0.0
                left_lim = 0.0

            # compute mean using cdf
            self.mean = integrate.quad(integrand, left_lim, self.b)[0] + correction
        else:
            # if no pdf or cdf is defined, return nan
            self.mean = float('NaN')
        return self.mean

    def compute_var(self):
        """
        Computes, sets and returns self.var = variance of the random variable.
        """
        # compute variance according to availability of pdf or cdf
        if hasattr(self, 'pdf'):
            # compute variance using pdf
            self.var = integrate.quad(lambda x: x*x*self.pdf(x), self.a, self.b)[0] - self.mean**2
        elif hasattr(self, 'cdf'):
            # parts of integrand
            plus = lambda x: 2.0*x*(1.0 - self.cdf(x))
            minus = lambda x: 2.0*x*self.cdf(x)

            # left limit of integration
            left_lim = self.a

            # decide integrand and correction term according to self.a and self.b being finite/infinite
            if not np.isinf(self.a):
                integrand = plus
                correction = self.a**2 - self.mean**2
            elif not np.isinf(self.b):
                integrand = minus
                correction = self.b**2 - self.mean**2
            else:
                integrand = lambda x: plus(x) - minus(-x)
                correction = - self.mean**2
                left_lim = 0.0

            # compute variance using cdf
            self.var = integrate.quad(integrand, left_lim, self.b)[0] + correction
        else:
            # if no pdf or cdf is defined, return nan
            self.var = float('NaN')
        return self.var

        def find_mean_var(self):
            """
            Sets self.mean and self.var using known formulae for known distributions.
            Returns results as a {'mean': -, 'var': -} dict.
            """
            if self.name == 'gamma':
                shape, scale = self.params['shape'], self.params['scale']
                self.mean = shape*scale
                self.var = shape*scale**2
            return {'mean': self.mean, 'var': self.var}

#############################################
# Sampling algorithm (function) definitions #
#############################################

def inverse_transform(inv_cdf, **params):
    """
    The inverse transform algorithm for sampling.
    Requires inverse of the cdf of the random variable to be sampled.
    params = dict of parameters of the distribution
    """
    return inv_cdf(np.random.uniform(), **params)

def composition(sim_components, probabilties):
    """
    The composition technique for sampling.
    Requires a list of simulations and a discrete probability distribution.
    """
    return sim_components[np.random.choice(len(sim_components), p = probabilties)].algorithm()

def rejection(target_rv, helper_rv, ratio_bound):
    """
    The accept-reject method for sampling.
    Requires a target and a helper random variable and an upper bound for the ratio of their pdfs.
    """
    while True:
        sample = helper_rv.algorithm()
        if np.random.uniform() <= target_rv.pdf(sample)/(ratio_bound*helper_rv.pdf(sample)):
            return sample


################################
# Simulation class definitions #
################################

class Simulation(object):
    """
    This is a class for simulating a random variable.
    It requires a target random variable and an algorithm to simulate it.
    """

    def __init__(self, target_rv, algorithm, **algorithm_args):
        """
        target_rv = random variable to simulate
        algorithm = a function that produces a single sample of target_rv
        algorithm_args = dict of keyword arguments that are passed to algorithm
        """
        # assign basic attributes
        self.rv = target_rv # random variable to simulate
        self.algorithm_args = algorithm_args # keyword arguments for self.algorithm which produces a single sample
        # self.uniform  = np.random.uniform # uniform distribution, needed for multiprocessing compatibility
        # self.choice = np.random.choice # discrete distribution, needed for multiprocessing compatibility
        # self.gamma = np.random.gamma # gamma distribution, needed for multiprocessing compatibility

        # set built-in algorithm simulation/sampling if possible
        if algorithm == 'inverse':
            self.algorithm = lambda *args: inverse_transform(self.algorithm_args['inv_cdf'], **self.rv.params) # algorithm_args = {'inv_cdf': -}
        elif algorithm == 'composition':
            self.algorithm = lambda *args: composition(**self.algorithm_args) # algorithm_args = {'sim_components': -, 'probabilties': -}
        elif algorithm == 'rejection':
            self.algorithm = lambda *args: rejection(self.rv, **self.algorithm_args) # algorithm_args = {'helper_rv': -, 'ratio_bound': -}
        else:
            self.algorithm = algorithm

    @timer
    def generate(self, sample_size, *args):
        """
        Generates a batch of samples using self.algorithm
        args are the arguments that are passed to self.algorithm ???
        """
        self.size = sample_size # number of samples to be collected
        self.samples = [self.algorithm(*args) for i in range(self.size)] # container for the collected samples
        self.mean = np.mean(self.samples)
        self.var = np.var(self.samples, ddof = 1) # unbiased estimator

    def compare(self, file_path = None, display = True, target_cdf_pts = 100, inf_limits = [-10.0, 10.0]):
        """
        Draws cdfs for target and simulation and sets self.ecdf.
        file_path is the location where the image file is saved, image file is not saved in case file_path = None (default).
        The plot is displayed on screen if display = True (default).
        target_cdf_pts is the number of points used to plot the target cdf.
        inf_limits is a finite interval for np.linspace in case rv.pdf has unbounded support.
        """
        # compute target mean and variance if not already computed
        if not hasattr(self.rv, 'mean'):
            self.rv.compute_mean()
        if not hasattr(self.rv, 'var'):
            self.rv.compute_var()

        # compute and plot simulated cdf
        self.ecdf = ECDF(self.samples)
        plt.figure(figsize = (7,6))
        plt.plot(self.ecdf.x, self.ecdf.y, label = 'simulation ($\mu$ = {:.4f}, $\sigma^2$ = {:.4f})'.format(self.mean, self.var))

        # fix limits for np.linspace in case rv.a or rv.b is unbounded
        left_lim = inf_limits[0] if np.isinf(self.rv.a) else self.rv.a
        right_lim = inf_limits[1] if np.isinf(self.rv.b) else self.rv.b

        # plot target cdf
        x = np.linspace(left_lim, right_lim, target_cdf_pts, endpoint = True)
        y = [self.rv.cdf(pt) for pt in x]
        plt.plot(x, y, label = 'target ($\mu$ = {:.4f}, $\sigma^2$ = {:.4f})'.format(self.rv.mean, self.rv.var))

        # write textual info on the plot
        plt.title('CDF vs ECDF')
        plt.xlabel('x')
        plt.legend()

        # save and display
        if file_path is not None:
            plt.savefig(file_path)
        if display:
            plt.show()
