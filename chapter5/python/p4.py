# This script requires 3 command line arguments (in this order) : alpha, beta, sample size
import sys
import numpy as np
from simulate import RVContinuous, Simulation

# unpack command-line arguments
alpha = float(sys.argv[1])
beta = float(sys.argv[2])
sample_size = int(sys.argv[3])

# target probability distribution
def target_cdf(x, alpha, beta):
    try:
        return 1.0 - np.exp(-alpha*x**beta)
    except:
        return 1.0

# inverse of the target distribution
def inv_cdf(y, alpha, beta):
    return (-np.log(1-y)/alpha)**(1.0/beta)

# simulate and compare
rv = RVContinuous(support = [0.0, np.inf], cdf = target_cdf, alpha = alpha, beta = beta)
sim = Simulation(target_rv = rv, algorithm = 'inverse', inv_cdf = inv_cdf)
sim.generate(sample_size)
sim.compare(file_path = '../images/p4_{}_{}_{}.png'.format(alpha, beta, sample_size), inf_limits = [0.0, 2.0])
