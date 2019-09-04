# This script requires 1 command line argument : sample size
import sys
import numpy as np
from simulate import RVContinuous, Simulation

# unpack command-line arguments
sample_size = int(sys.argv[1])

# target probabilty distribution
def target_cdf(x):
    if x < 1.0:
        return (1- np.exp(-2*x) + 2*x)/3.0
    else:
        return (3 - np.exp(-2*x))/3.0

# cdfs for the component distributions
cdfs = [lambda x, *args: 1.0 - np.exp(-2*x), lambda x, *args: x if x < 1.0 else 1.0]

# inverse for the component distributions
inv_cdfs = [lambda y, *args: -0.5*np.log(1-y), lambda y, *args: y]

# simulate and compare
rv = RVContinuous(support = [0.0, np.inf], cdf = target_cdf)
sim_components = [Simulation(RVContinuous(support = [0.0, np.inf], cdf = cdf), algorithm = 'inverse', inv_cdf = inv_cdf) for cdf, inv_cdf in zip(cdfs, inv_cdfs)]
sim = Simulation(target_rv = rv, algorithm = 'composition', sim_components = sim_components, probabilties = [1.0/3.0, 2.0/3.0])
sim.generate(sample_size)
sim.compare(file_path = '../images/p8b_{}.png'.format(sample_size))
