import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import rv_continuous
from statsmodels.distributions.empirical_distribution import ECDF
from random import uniform
from utility import timer

# A class for defining generic continuous random variables
class RVContinuous(rv_continuous):
    def __init__(self, pdf):
        self.pdf = pdf
    def _pdf(self, x, *args):
        return self.pdf(x, *args)

# A bare-bones base class for algorithms that simulate a random variable
class Simulation(object):
    def __init__(self):
        self.method = None # method of simulation/sampling
        self.uniform  = uniform

    # generates samples using self.method
    @timer
    def generate(self, sample_size, *args):
        self.size = sample_size # number of samples to be collected
        self.samples = [self.method(*args) for i in range(self.size)] # container for the collected samples
        self.mean = np.mean(self.samples)
        self.var = np.var(self.samples, ddof = 1)

    # draws density curves for target and simulation
    def draw(self, target, range_, pts = 100, file_path = None, display = False):
        plt.figure(figsize=(7,6))
        sns.set()
        sns.kdeplot(np.array(self.samples), label = 'simulation ($\mu$ = {:.4f}, $\sigma^2$ = {:.4f})'.format(self.mean, self.var))
        x = np.linspace(range_[0], range_[1], pts, endpoint = True)
        y = [target(pt) for pt in x]
        plt.plot(x, y, label = 'target ($\mu$ = {:.4f}, $\sigma^2$ = {:.4f})'.format(self.target_mean, self.target_var))
        plt.title('Density plots')
        plt.xlabel('x')
        plt.legend()
        if file_path is not None:
            plt.savefig(file_path)
        if display is True:
            plt.show()

    # draws distribution curves for target and simulation
    def draw_cdf(self, target, range_, pts = 100, file_path = None, display = False):
        plt.figure(figsize = (7,6))
        self.ecdf = ECDF(self.samples)
        plt.plot(self.ecdf.x, self.ecdf.y, label = 'simulation ($\mu$ = {:.4f}, $\sigma^2$ = {:.4f})'.format(self.mean, self.var))
        x = np.linspace(range_[0], range_[1], pts, endpoint = True)
        y = [target(pt) for pt in x]
        plt.plot(x, y, label = 'target ($\mu$ = {:.4f}, $\sigma^2$ = {:.4f})'.format(self.target_mean, self.target_var))
        plt.title('CDF vs ECDF')
        plt.xlabel('x')
        plt.legend()
        if file_path is not None:
            plt.savefig(file_path)
        if display is True:
            plt.show()

    # generates a plot juxtaposing the target with the simulated distribution
    @timer
    def visualize(self, target_density, range_, pts = 100, file_path = None, display = False):
        # Base class for a custom continuous random variable
        self.rv = RVContinuous(target_density) # creates the target random variable
        self.target_mean = self.rv.mean()
        self.target_var = self.rv.var()
        self.draw(target_density, range_, pts, file_path, display)


    # generates a plot juxtaposing the target with the simulated distribution given the mean and variance of the target distribution
    # suffix 'man' indicates you have to supply target_mean, target_dist manually to the function
    @timer
    def vis_man(self, target_density, range_, target_mean, target_var, pts = 100, file_path = None, display = False):
        self.target_mean = target_mean
        self.target_var = target_var
        self.draw(target_density, range_, pts, file_path, display)

    # generates a plot juxtaposing the target with the simulated distribution given the mean and variance of the target distribution
    @timer
    def vis_cdf_man(self, target_dist, range_, target_mean, target_var, pts = 100, file_path = None, display = False):
        self.target_mean = target_mean
        self.target_var = target_var
        self.draw_cdf(target_dist, range_, pts, file_path, display)

# Implements the inverse transform algorithm
class InverseTransform(Simulation):
    def __init__(self, inv_dist):
        Simulation.__init__(self)
        self.method = lambda *args: inv_dist(self.uniform(0.0, 1.0))

# Implements composition method for sampling
class InverseComposition(Simulation):
    def __init__(self, inv_dist_list, probabilties):
        Simulation.__init__(self)
        self.inv_dist_list = inv_dist_list
        self.probabilties = probabilties
        self.method = lambda i, *args: self.inv_dist_list[i](self.uniform(0.0, 1.0))

    # generates samples using self.method
    @timer
    def generate(self, sample_size):
        self.size = sample_size # number of samples to be collected
        self.samples = [self.method(i) for i in np.random.choice(len(self.inv_dist_list), self.size, self.probabilties)] # container for the collected samples
        self.mean = np.mean(self.samples)
        self.var = np.var(self.samples, ddof = 1)

# Implements accept-reject method for sampling
class AcceptReject(Simulation):
    def __init__(self, target_dist, helper_rv, ratio_bound):
        self.helper_rv = helper_rv
        self.ratio_bound = ratio_bound

    def method(self, helper_args = (), *args):
        rv_obtained = False
        while rv_obtained is False:
            sample = self.helper_rv.method(*helper_args)
            if self.uniform(0.0, 1.0) <= self./(self.ratio_bound*self.helper_rv.density(sample))
