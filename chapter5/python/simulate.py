import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from scipy.stats import rv_continuous
from statsmodels.distributions.empirical_distribution import ECDF
from random import uniform
from utility import timer

# A class for defining generic continuous random variables
class RVContinuous(object):
    # requires pdf or cdf of the random variable
    def __init__(self, support, cdf = None, pdf = None, inv_cdf = None, params = None):
        # assign basic attributes
        self.a = support[0] # left endpoint of support interval of pdf
        self.b = support[1] # right endpoint of support interval of pdf
        self.params = params # parameters of pdf and cdf
        if pdf is not None:
            self.pdf_ = pdf # family of pdfs without parmeters specified
            self.pdf = lambda x: self.pdf_(x, self.params) # pdf with parameters specified
        if cdf is not None:
            self.cdf_ = cdf # family of cdfs without parmeters specified
            self.cdf = lambda x: self.cdf_(x, self.params) # cdf with parameters specified
        if inv_cdf is not None:
            self.inv_cdf_ = inv_cdf # family of inverse cdfs without parmeters specified
            self.inv_cdf = lambda x: self.inv_cdf_(x, self.params) # inverse cdf with parameters specified

    # reset parameters of the distribution
    def set_params(self, new_params):
        self.params = new_params
        if hasattr(self, 'pdf'):
            self.pdf = lambda x: self.pdf_(x, self.params)
        if hasattr(self, 'cdf'):
            self.cdf = lambda x: self.cdf_(x, self.params)
        if hasattr(self, 'inv_cdf'):
            self.inv_cdf = lambda x: self.inv_cdf_(x, self.params)

    # returns and sets self.mean
    def compute_mean(self):
        # compute mean according to availability of pdf or cdf
        if hasattr(self, 'pdf'):
            # compute mean using pdf
            self.mean = integrate.quad(lambda x: x*self.pdf(x), self.a, self.b)[0]
        elif hasattr(self, 'cdf'):
            # parts of integrand
            plus = lambda x: 1.0 - self.cdf(x)
            minus = lambda x: -self.cdf(x)

            # range of integration
            left_lim = self.a
            right_lim = self.b

            # decide integrand and correction term according to self.a and self.b being finite/infinite
            if np.isinf(self.a) is False:
                integrand = plus
                correction = self.a
            elif np.isinf(self.b) is False:
                integrand = minus
                correction = self.b
            else:
                integrand = lambda x: plus(x) + minus(-x)
                correction = 0.0
                left_lim = 0.0

            # compute mean using cdf
            self.mean = integrate.quad(integrand, left_lim, right_lim)[0] + correction
        else:
            # if no pdf or cdf is defined, return nan
            self.mean = float('NaN')
        return self.mean

    # returns and sets self.var
    def compute_variance(self):
        # compute variance according to availability of pdf or cdf
        if hasattr(self, 'pdf'):
            # compute variance using pdf
            self.var = integrate.quad(lambda x: x*x*self.pdf(x), self.a, self.b)[0] - self.mean**2
        elif hasattr(self, 'cdf'):
            # parts of integrand
            plus = lambda x: 2.0*x*(1.0 - self.cdf(x))
            minus = lambda x: 2.0*x*self.cdf(x)

            # range of integration
            left_lim = self.a
            right_lim = self.b

            # decide integrand and correction term according to self.a and self.b being finite/infinite
            if np.isinf(self.a) is False:
                integrand = plus
                correction = self.a**2 - self.mean**2
            elif np.isinf(self.b) is False:
                integrand = minus
                correction = self.b**2 - self.mean**2
            else:
                integrand = lambda x: plus(x) - minus(-x)
                correction = - self.mean**2
                left_lim = 0.0

            # compute variance using cdf
            self.var = integrate.quad(integrand, left_lim, right_lim)[0] + correction
        else:
            # if no pdf or cdf is defined, return nan
            self.var = float('NaN')
        return self.var

# A bare-bones base class for algorithms that simulate a random variable
class Simulation(object):
    # requires target RVContinuous object with cdf attribute
    def __init__(self, target_random_variable):
        # assign basic attributes
        self.rv = target_random_variable # random variable to simulate
        self.algorithm = None # algorithm for simulation/sampling
        self.uniform  = uniform # uniform distribution from random module, needed for multiprocessing compatibility

    # generates samples using self.algorithm
    @timer
    def generate(self, sample_size, *args):
        self.size = sample_size # number of samples to be collected
        self.samples = [self.algorithm(*args) for i in range(self.size)] # container for the collected samples
        self.mean = np.mean(self.samples)
        self.var = np.var(self.samples, ddof = 1) # unbiased estimator

    # draws cdfs for target and simulation and sets self.ecdf
    def compare(self, file_path = None, display = True, target_cdf_pts = 100):
        # compute target mean and variance if not already computed
        if hasattr(self.rv, 'mean') is False:
            self.rv.compute_mean()
        if hasattr(self.rv, 'var') is False:
            self.rv.compute_variance()

        # compute and plot simulated cdf
        self.ecdf = ECDF(self.samples)
        plt.figure(figsize = (7,6))
        plt.plot(self.ecdf.x, self.ecdf.y, label = 'simulation ($\mu$ = {:.4f}, $\sigma^2$ = {:.4f})'.format(self.mean, self.var))

        # plot target cdf
        x = np.linspace(self.rv.a, self.rv.b, target_cdf_pts, endpoint = True)
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

# Implements the inverse transform algorithm
class InverseTransform(Simulation):
    # requires target_random_variable to have inv_cdf attribute
    def __init__(self, target_random_variable):
        # inherit attributes from Simulation
        Simulation.__init__(self, target_random_variable)
        # sampling algorithm
        self.algorithm = lambda *args: self.rv.inv_cdf(self.uniform(0.0, 1.0))
"""
# Implements composition algorithm for sampling
class InverseComposition(Simulation):
    def __init__(self, rv_list, probabilties):
        Simulation.__init__(self)
        self.inv_dist_list = inv_dist_list
        self.probabilties = probabilties
        self.algorithm = lambda i, *args: self.inv_dist_list[i](self.uniform(0.0, 1.0), *args)

    # generates samples using self.algorithm
    @timer
    def generate(self, sample_size):
        self.size = sample_size # number of samples to be collected
        self.samples = [self.algorithm(i) for i in np.random.choice(len(self.inv_dist_list), self.size, self.probabilties)] # container for the collected samples
        self.mean = np.mean(self.samples)
        self.var = np.var(self.samples, ddof = 1)

# Implements accept-reject algorithm for sampling
class AcceptReject(Simulation):
    def __init__(self, target_rv, helper_rv, ratio_bound):
        self.target_rv = target_rv
        self.helper_rv = helper_rv
        self.ratio_bound = ratio_bound

    def algorithm(self, *args):
        while True:
            sample = self.helper_rv.algorithm(*helper_args)
            if self.uniform(0.0, 1.0) <= self.target_rv.pdf(sample)/(self.ratio_bound*self.helper_rv.pdf(sample)):
                return sample
"""
