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

    def __init__(self, name = 'unknown', support = (0.0, 1.0), cdf = None, pdf = None, find_mean = None, find_var = None, **params):
        """
        ---------
        Arguments
        ---------
        name = name of the random variable
        support = support of the pdf, default = (0.0, 1.0)
        find_mean = custom function for computing mean, accpets parameters of the distribution as **kwargs
        find_var = custom function for computing variance, accpets parameters of the distribution as **kwargs
        params = dict of keyword arguments that are passed to pdf, cdf and inv_cdf

        -----
        Notes
        -----
        Either pdf or cdf is required for mean and variance computation. One of them can be omitted.
        In case the random variable has a well-known distribution, providing the name of the random variable and
        **params = parameters of the distribution will set all other arguments automatically.
        Currently a known name can be anything in the list ['gamma']. Dafault is 'unknown'.
        """
        # set support and pdf/cdf for known distributions
        if name == 'gamma':
            support = (0.0, np.inf)
            cdf = lambda x, shape, scale: gamma.cdf(x, shape, scale = scale)
            pdf = lambda x, shape, scale: gamma.pdf(x, shape, scale = scale)

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
        if find_mean is not None:
            self.find_mean_ = find_mean # family of find_means without parmeters specified
            self.find_mean = lambda: self.find_mean_(**self.params) # find_mean with parameters specified
        if find_var is not None:
            self.find_var_ = find_var # family of find_vars without parmeters specified
            self.find_var = lambda: self.find_var_(**self.params) # find_var with parameters specified)
        self.mean = 'not_yet_computed'
        self.var = 'not_yet_computed'


    def set_params(self, **new_params):
        """
        -----------
        Description
        -----------
        Resets parameters of the distribution to new_params.
        Passing only the parameters that need to be changed suffices.

        ------
        Return
        ------
        None
        """
        for key, value in new_params.items():
            self.params[key] = value
        if hasattr(self, 'pdf'):
            self.pdf = lambda x: self.pdf_(x, **self.params)
        if hasattr(self, 'cdf'):
            self.cdf = lambda x: self.cdf_(x, **self.params)
        if hasattr(self, 'find_mean'):
            self.find_mean = lambda: self.find_mean_(**self.params)
        if hasattr(self, 'find_var'):
            self.find_var = lambda: self.find_var_(**self.params)


    def compute_mean(self):
        """
        -----------
        Description
        -----------
        Computes and sets self.mean = expected value of the random variable.

        ------
        Return
        ------
        None
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
        -----------
        Description
        -----------
        Computes and sets self.var = variance of the random variable.

        ------
        Return
        ------
        None
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

    def set_stats(self, stats = ()):
        """
        -----------
        Description
        -----------
        Computes and sets the user-chosen statistics of the distribution using the easiest possible methods
        depending on availability of find_mean, find_var etc.

        ---------
        Arguments
        ---------
        stats = list/tuple of statistic names to be computed.

        ------
        Return
        ------
        None

        -----
        Notes
        -----
        If the value is set to True, set_stats will try to compute the corresponding statistic.
        If stats = () (default), all statistics are computed.
        """
        for stat in stats:
            if hasattr(self, 'find_' + stat):
                setattr(self, stat, getattr(self, 'find_' + stat)())
            else:
                setattr(self, stat, getattr(self, 'compute_' + stat)())

    def set_unset_stats(self, stats = ()):
        """
        -----------
        Description
        -----------
        Sets only unset statistics of the distribution using self.set_stats.

        ---------
        Arguments
        ---------
        stats = list/tuple of unset statistics.
        In case stats = () (default), all unset statistics are set.

        ------
        Return
        ------
        None
        """
        if stats == ():
            stats = ('mean', 'var')
        stats_to_compute = []
        for stat in stats:
            if hasattr(self, stat):
                stats_to_compute.append(stat)
        self.set_stats(stats_to_compute)


#############################################
# Sampling algorithm (function) definitions #
#############################################

def inverse_transform(inv_cdf, **params):
    """
    -----------
    Description
    -----------
    The inverse transform algorithm for sampling.

    ---------
    Arguments
    ---------
    inv_cdf = inverse of the cdf of the random variable to be sampled
    params = dict of parameters of the distribution

    ------
    Return
    ------
    Generated sample
    """
    return inv_cdf(np.random.uniform(), **params)

def composition(sim_components, probabilties):
    """
    -----------
    Description
    -----------
    The composition technique for sampling.

    ---------
    Arguments
    ---------
    sim_components = list of simulations
    probabilties = a discrete probability distribution

    ------
    Return
    ------
    Generated sample
    """
    return sim_components[np.random.choice(len(sim_components), p = probabilties)].algorithm()

def rejection(target_rv, helper_sim, ratio_bound):
    """
    -----------
    Description
    -----------
    The accept-reject method for sampling.

    ---------
    Agruments
    ---------
    target_rv = target random variable.
    helper_sim = simulation for helper random variable with pdf assigned.
    ratio_bound = an upper bound for the ratio of the pdfs.

    ------
    Return
    ------
    Generated sample
    """
    while True:
        sample = helper_sim.algorithm()
        if np.random.uniform() <= target_rv.pdf(sample)/(ratio_bound*helper_sim.rv.pdf(sample)):
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
        ---------
        Arguments
        ---------
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
        self.set_algorithm(algorithm, **algorithm_args)

    @timer
    def generate(self, sample_size):
        """
        -----------
        Description
        -----------
        Generates a batch of samples using self.algorithm
        args are the arguments that are passed to self.algorithm ???

        ------
        Return
        ------
        None
        """
        self.size = sample_size # number of samples to be collected
        self.samples = [self.algorithm() for i in range(self.size)] # container for the collected samples
        self.mean = np.mean(self.samples)
        self.var = np.var(self.samples, ddof = 1) # unbiased estimator

    def compare(self, file_path = None, display = True, target_cdf_pts = 100):
        """
        -----------
        Description
        -----------
        Draws cdfs for target and simulation and sets self.ecdf.

        ---------
        Arguments
        ---------
        file_path is the location where the image file is saved, image file is not saved in case file_path = None (default).
        The plot is displayed on screen if display = True (default).
        target_cdf_pts is the number of points used to plot the target cdf.

        ------
        Return
        ------
        Figure and axes objects for the generated plot (in this order)
        """
        # compute target mean and variance if not already computed
        self.rv.set_unset_stats(('mean', 'var'))

        # compute and plot simulated cdf
        self.ecdf = ECDF(self.samples)
        fig = plt.figure(figsize = (7,6))
        ax = fig.add_subplot(111)
        ax.plot(self.ecdf.x, self.ecdf.y, label = 'simulation ($\mu$ = {:.4f}, $\sigma^2$ = {:.4f})'.format(self.mean, self.var))

        # fix limits for np.linspace in case rv.a or rv.b is unbounded
        left_lim = self.ecdf.x[1] if np.isinf(self.rv.a) else self.rv.a
        right_lim = self.ecdf.x[-1] if np.isinf(self.rv.b) else self.rv.b

        # plot target cdf
        x = np.linspace(left_lim, right_lim, target_cdf_pts, endpoint = True)
        y = [self.rv.cdf(pt) for pt in x]
        ax.plot(x, y, label = 'target ($\mu$ = {:.4f}, $\sigma^2$ = {:.4f})'.format(self.rv.mean, self.rv.var))

        # write textual info on the plot
        ax.set_title('CDF vs ECDF')
        ax.set_xlabel('x')
        ax.legend()

        # save and display
        if file_path is not None:
            fig.savefig(file_path)
        if display:
            plt.show()
        return fig, ax

    def set_algorithm(self, algorithm, **algorithm_args):
        """
        -----------
        Description
        -----------
        Sets self.algorithm

        ---------
        Arguments
        ---------
        algorithm = a function that produces a single sample of target_rv
        algorithm_args = dict of keyword arguments that are passed to algorithm

        ------
        Return
        ------
        None
        """
        # set built-in algorithm for simulation/sampling if possible
        if algorithm == 'inverse':
            algorithm = lambda *args: inverse_transform(self.algorithm_args['inv_cdf'], **self.rv.params) # algorithm_args = {'inv_cdf': -}
        elif algorithm == 'composition':
            algorithm = lambda *args: composition(**self.algorithm_args) # algorithm_args = {'sim_components': -, 'probabilties': -}
        elif algorithm == 'rejection':
            algorithm = lambda *args: rejection(self.rv, **self.algorithm_args) # algorithm_args = {'helper_rv': -, 'ratio_bound': -} (helper_rv must have pdf assigned)
        elif algorithm == 'gamma':
            algorithm = lambda *args: np.random.gamma(**self.rv.params)

        def algorithm_(*args):
            self.current_value = algorithm(*args)
            return self.current_value

        self.algorithm = lambda *args: algorithm_(*args)



########################################
# Stochastic Process class definitions #
########################################

class StochasticProcess(object):
        """
        This is a class for defining generic continuous-time stochastic processes.
        """

        def __init__(self, support = (0.0, 1.0), cdf = None, pdf = None, find_mean = None, find_var = None, **params):
            """
            ---------
            Arguments
            ---------
            support = support of the pdf, default = (0.0, 1.0)
            find_mean = custom function for computing mean, accpets parameters of the distribution as **kwargs
            find_var = custom function for computing variance, accpets parameters of the distribution as **kwargs
            params = dict of keyword arguments that are passed to pdf, cdf and inv_cdf

            -----
            Notes
            -----
            Either pdf or cdf is required for mean and variance computation. One of them can be omitted.
            In case the random variable has a well-known distribution, providing the name of the random variable and
            **params = parameters of the distribution will set all other arguments automatically.
            """
            # assign basic attributes
            self.name = name # name of the random variable
            self.a, self.b = support # left and right endpoints of support interval of pdf
            self.params = params # parameters of pdf and cdf
            if pdf is not None:
                self.pdf_ = pdf # family of pdfs without parmeters specified
                self.pdf = lambda x, t: self.pdf_(x, t, **self.params) # pdf with parameters specified
            if cdf is not None:
                self.cdf_ = cdf # family of cdfs without parmeters specified
                self.cdf = lambda x: self.cdf_(x, t, **self.params) # cdf with parameters specified
            if find_mean is not None:
                self.find_mean_ = find_mean # family of find_means without parmeters specified
                self.find_mean = lambda t: self.find_mean_(t, **self.params) # find_mean with parameters specified
            if find_var is not None:
                self.find_var_ = find_var # family of find_vars without parmeters specified
                self.find_var = lambda t: self.find_var_(t, **self.params) # find_var with parameters specified)

        def get_rv(self, t):
            """
            -----------
            Description
            -----------
            Generates the random variable at time t

            ---------
            Arguments
            ---------
            t = time at which the random variable is to be generated

            ------
            Return
            ------
            Generated random variable
            """
            # generate arguments for the constructor of RVContinuous
            if hasattr(self, 'pdf_'):
                pdf = lambda x: self.pdf_(x, t)
            else:
                pdf = None
            if hasattr(self, 'cdf_'):
                cdf = lambda x: self.cdf_(x, t)
            else:
                cdf = None
            if hasattr(self, 'find_mean_'):
                find_mean = lambda: self.find_mean_(t)
            else:
                find_mean = None
            if hasattr(self, 'find_var_'):
                find_var = lambda: self.find_var_(t)
            else:
                find_var = None
            return RVContinuous(support = [self.a, self.b], cdf = cdf, pdf = pdf, find_mean = find_mean, find_var = find_var, **self.params)

        def generate_paths(self, interval, num_paths = 1):
            """
            -----------
            Description
            -----------
            Generates sample paths

            ---------
            Arguments
            ---------
            interval = time domain of the sample paths

            ------
            Return
            ------
            None
            """
            pass
