# This script requires 3 command line arguments (in this order) : alpha, beta, sample size
import sys
import numpy as np
from simulate import InverseTransform, RVContinuous

# target probability distribution
def target_cdf(x, params):
    alpha_, beta_ = params
    try:
        return 1.0 - np.exp(-alpha_*x**beta_)
    except:
        return 1.0

# inverse of the target distribution
def inv_cdf(y, params):
    alpha_, beta_ = params
    return (-np.log(1-y)/alpha_)**(1.0/beta_)

# set parameter values for cdf
alpha, beta = float(sys.argv[1]), float(sys.argv[2])

# simulate and compare
sim = InverseTransform(RVContinuous(support = [0.0, np.inf], cdf = target_cdf, inv_cdf = inv_cdf, params = [alpha, beta]))
sim.generate(int(sys.argv[3]))
sim.compare(file_path = '../images/p4_{}_{}_{}.png'.format(sys.argv[1], sys.argv[2], sys.argv[3]), inf_limits = [0.0, 2.0])
