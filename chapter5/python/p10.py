# This script requires 1 command line argument: sample size
import sys
import numpy as np
import scipy.stats as ss
from simulate import Simulation

# Implements gamma distribution
class Gamma(Simulation):
    def __init__(self, n, l):
        Simulation.__init__(self)
        self.n = n # parameter n for gamma distribution
        self.l = float(l) # parameter lambda for gamma distribution
        self.method = lambda: -np.log(np.prod(np.random.uniform(0.0, 1.0, self.n)))/self.l

gamma = Gamma(1000, 1/800.0)
gamma.generate(int(sys.argv[1]))
target_mean = gamma.n/gamma.l
target_var = gamma.n/gamma.l**2
target_dist = ss.gamma(1000, scale = 1/800.0).cdf
gamma.vis_cdf_man(target_dist, [0.0, 10.0], target_mean, target_var, file_path = '../images/p10_dist_{}_{}.png'.format('incomp', sys.argv[1]), display = True)
print("Simulated probabilty = {}\nTarget probabilty = {}".format(1 - gamma.ecdf(1e6), 1 - target_dist(1e6)))
