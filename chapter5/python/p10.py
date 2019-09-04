# This script requires 1 command line argument: sample size
import sys
from simulate import Composition, RVContinuous
from scipy.stats import binom, gamma

# unpack command-line arguments
sample_size = int(sys.argv[1])

# problem parameters
n = 1000 # number of pocilyholders
p = 0.05 # probabilty of presenting a claim
mu = 500 # expected amount of claim
A = 50000 # least total amount according to the problem

# gamma distribution
def gamma_(x, params):
    k, theta = params
    return gamma.pdf(x/theta, k)/theta

# target probability distribution
def target_cdf(x, params):
    k, theta, n = params
    return sum((binom.pmf(k, n, p)*gamma_(x, [k, theta]) for k in range(1, n+1)))

# simulate and compare
rv_gamma = RVGamma(k = 1, theta = mu)
sim_components = []
for i in range(n):
    sim_components.append(
