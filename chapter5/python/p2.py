# This script requires 1 command line argument : sample size
import sys
from simulate import InverseTransform, RVContinuous

# target probability density
def target_pdf(x, *args):
    if x >= 2.0 and x <= 3.0:
        return (x-2)/2.0
    elif x > 3.0 and x <= 6.0:
        return (6-x)/6.0
    else:
        return 0.0

# create a continuous random variable
rv = RVContinuous([2,6], pdf = target_pdf)
print(rv.compute_mean(), rv.compute_variance())

"""# inverse of the target distribution
def inv_dist(y, *args):
    if y <= 0.25:
        return 2*(y**0.5 + 1)
    else:
        return 6 - 2*(3*(1-y))**0.5

it = InverseTransform(inv_dist)
it.generate(int(sys.argv[1]))
target_mean = (1.0 - 1.05*np.exp(-0.05))/c
target_var =  (2.0 -(0.05*2.05+2.0)*np.exp(-0.05))/c - target_mean**2
it.vis_cdf_man(target_dist, [2.0, 6.0], target_mean, target_var, file_path = '../images/p2_dist_{}_{}.png'.format('it', sample_size), display = True)
"""
