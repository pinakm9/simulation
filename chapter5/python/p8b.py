# This script requires 1 command line argument : sample size
import sys
import numpy as np
from simulate import InverseComposition

# target probabilty distribution
def target_dist(x):
    if x < 1.0:
        return (1- np.exp(-2*x) + 2*x)/3.0
    else:
        return (3 - np.exp(-2*x))/3.0

# inverse for the distribution F_1
def inv_1(y):
    return -0.5*np.log(1-y)

# inverse for the distribution F_1
def inv_2(y):
    return y

comp = InverseComposition([inv_1, inv_2], [1/3.0, 2/3.0])
comp.generate(int(sys.argv[1]))
target_mean = 0.5
target_var =  7/18.0 - target_mean**2
comp.vis_cdf_man(target_dist, [0.0, 10.0], target_mean, target_var, file_path = '../images/p8b_dist_{}_{}.png'.format('incomp', sys.argv[1]), display = True)
