# This script requires 1 command line argument: sample size
import sys
from simulate import RVContinuous, Simulation
import numpy as np

# unpack command-line arguments
sample_size = int(sys.argv[1])

# target probability distribution
def target_cdf(x):
    return 1.0 - np.exp(-x) - np.exp(-2.0*x) + np.exp(-3.0*x)

# target probability density
def target_pdf(x):
    return np.exp(-x) + 2.0*np.exp(-2.0*x) - 3.0*np.exp(-3.0*x)

# define our random variable to be simulated
rv = RVContinuous(support = [0.0, np.inf], cdf = target_cdf, pdf = target_pdf)

# fisrt simulation
helper_rv =  RVContinuous(support = [0.0, np.inf], pdf = lambda x: np.exp(-x))
helper_sim = Simulation(target_rv = helper_rv, algorithm = lambda *args: np.random.exponential())
sim = Simulation(target_rv = rv, algorithm = 'rejection', helper_sim = helper_sim, ratio_bound = 4.0)
sim.generate(sample_size)
sim.compare(file_path = '../images/p16_1_{}.png'.format(sample_size))

# second simulation
helper_rv =  RVContinuous(support = [0.0, np.inf], pdf = lambda x: 1.0)
helper_sim = Simulation(target_rv = helper_rv, algorithm = lambda *args: np.random.exponential())
sim = Simulation(target_rv = rv, algorithm = 'rejection', helper_sim = helper_sim, ratio_bound = 4.0)
sim.generate(sample_size)
sim.compare(file_path = '../images/p16_2_{}.png'.format(sample_size))
