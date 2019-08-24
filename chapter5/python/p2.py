# This script requires 1 command line argument : sample size
import sys
from simulate import InverseTransform

# target probability density
def target_density(x):
    if x >= 2.0 and x <= 3.0:
        return (x-2)/2.0
    elif x > 3 and x <= 6.0:
        return (6-x)/6.0
    else:
        return 0.0

# inverse of the target distribution
def inv_dist(y):
    if y <= 0.25:
        return 2*(y**0.5 + 1)
    else:
        return 6 - 2*(3*(1-y))**0.5

it = InverseTransform(inv_dist)
it.generate(int(sys.argv[1]))
it.visualize(target_density, [2, 6], file_path = '../images/p2_density_{}_{}.png'.format('it', sys.argv[1]), display = True)
