# This script requires 1 command line argument : sample size
import sys
import numpy as np
import functools
from simulate import InverseComposition

# target probability distribution
def target_dist(x):
    return (x + x**3 + x**5)/3.0

# inverse for the distribution x^n
def inv_dist(n, y):
    return y**(1.0/n)

inv_dist_list = [functools.partial(inv_dist, n) for n in range(1,6,2)]
comp = InverseComposition(inv_dist_list, [1/3.0]*3)
comp.generate(int(sys.argv[1]))
target_mean = (1/2.0 + 3/4.0 + 5/6.0)/3.0
target_var =  (1/3.0 + 3/5.0 + 5/7.0)/3.0 - target_mean**2
comp.vis_cdf_man(target_dist, [0.0, 1.0], target_mean, target_var, file_path = '../images/p8a_dist_{}_{}.png'.format('incomp', sys.argv[1]), display = True)
