# This script requires 2 command line arguments (in this order) : sample size, degree of the target polynomial distribution
import sys
import numpy as np
import functools
from simulate import InverseComposition

# generate probability weights
degree = int(sys.argv[2])
alpha = np.random.random(size = degree)
alpha /= alpha.sum()

# target probability distribution
def target_dist(x):
    return sum([a*x**(i+1) for i, a in enumerate(alpha)])

# inverse for the distribution x^n
def inv_dist(n, y):
    return y**(1.0/n)

inv_dist_list = [functools.partial(inv_dist, n+1) for n in range(degree)]
comp = InverseComposition(inv_dist_list, list(alpha))
comp.generate(1500)
target_mean = sum([(i+1)*a/float(i+2) for i, a in enumerate(alpha)])
target_var =  sum([(i+1)*a/float(i+3) for i, a in enumerate(alpha)]) - target_mean**2
comp.vis_cdf_man(target_dist, [0.0, 1.0], target_mean, target_var, file_path = '../images/p8c_dist_{}_{}.png'.format('incomp', sys.argv[1]), display = True)
print('The generated probability weights are:\n{}'.format(alpha))
