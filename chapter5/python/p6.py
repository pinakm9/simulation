import numpy as np
from simulate import InverseTransform

c = 1.0-np.exp(-0.05)
# target probability distribution
def target_dist(x):
    return (1.0 - np.exp(-x))/c

# inverse of the target distribution
def inv_dist(y):
    return -np.log(1.0-c*y)

it = InverseTransform(inv_dist)
sample_size = 1000
it.generate(sample_size)
target_mean = (1.0 - 1.05*np.exp(-0.05))/c
target_var =  (2.0 -(0.05*2.05+2.0)*np.exp(-0.05))/c - target_mean**2
it.vis_cdf_man(target_dist, [0, 0.05], target_mean, target_var, file_path = '../images/p6_dist_{}_{}.png'.format('it', sample_size), display = True)
print('simulated mean = {:.4f}\ntarget mean = {:.4f}'.format(it.mean, it.target_mean))
