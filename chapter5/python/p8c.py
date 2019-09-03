# This script requires 2 command line arguments (in this order) : degree of the target polynomial distribution, sample size
import sys
import numpy as np
from simulate import InverseComposition, RVContinuous

# unpack command-line arguments
degree = int(sys.argv[1])
sample_size = int(sys.argv[2])

# generate probability weights (stored in alpha)
alpha = np.random.random(size = degree)
alpha /= alpha.sum()

# target probability distribution
def target_cdf(x, *args):
    return sum([a*x**(i+1) for i, a in enumerate(alpha)])

# distribution for x^n
def cdf(y, n):
    return x**n

# inverse for the distribution x^n
def inv_cdf(y, n):
    return y**(1.0/n)

# simulate and compare
rv_components = [RVContinuous(support = [0.0, 1.0], cdf = cdf, inv_cdf = inv_cdf, params = n+1) for n in range(degree)]
sim = InverseComposition(RVContinuous(support = [0.0, 1.0], cdf = target_cdf), rv_components, list(alpha))
sim.generate(sample_size)
sim.compare(file_path = '../images/p8c_{}_{}.png'.format(degree, sample_size))
print('The generated probability weights are:\n{}'.format(alpha))
