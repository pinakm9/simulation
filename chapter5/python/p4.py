# This script requires 3 command line arguments (in this order) : alpha, beta, sample size
import sys
import numpy as np
from simulate import InverseTransform
from math import gamma

# set parameter values
alpha, beta = float(sys.argv[1]), float(sys.argv[2])

# target probability density
def target_density(x):
    try:
        return alpha*beta*np.exp(-alpha*x**beta)*x**(beta-1)
    except OverflowError as Error:
        return 0.0
# inverse of the target distribution
def inv_dist(y):
    return (-np.log(1-y)/alpha)**(1.0/beta)

c = alpha**(-1.0/beta)
target_mean = c*gamma(1.0 + 1.0/beta)
target_var = c**2*(gamma(2.0 + 1.0/beta)-gamma(1.0 + 1.0/beta)**2)
it = InverseTransform(inv_dist)
it.generate(int(sys.argv[3]))
it.vis_man(target_density, [0, 10], target_mean, target_var, file_path = '../images/p4_density_{}_{}_{}_{}.png'.format('it', sys.argv[1], sys.argv[2], sys.argv[3]), display = True)
