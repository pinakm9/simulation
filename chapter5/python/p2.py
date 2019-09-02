# This script requires 1 command line argument : sample size
import sys
from simulate import InverseTransform, RVContinuous

# target probability density
def target_cdf(x, *args):
    if x >= 2.0 and x <= 3.0:
        return 0.25 * (x-2.0)**2
    elif x > 3.0 and x <= 6.0:
        return 0.25 + (x-3.0)*(9.0-x)/12.0
    else:
        return 0.0

# inverse of the target distribution
def inv_cdf(y, *args):
    if y <= 0.25:
        return 2*(y**0.5 + 1)
    else:
        return 6 - 2*(3*(1-y))**0.5

# create our target continuous random variable
rv = RVContinuous([2.0, 6.0], target_cdf, inv_cdf = inv_cdf)
print(rv.compute_mean(), rv.compute_variance())

# simulate and compare
sim = InverseTransform(rv)
sim.generate(int(sys.argv[1]))
sim.compare(file_path = '../images/p2_{}.png'.format(sys.argv[1]))
