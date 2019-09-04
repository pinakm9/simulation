# This script requires 1 command line argument : sample size
import sys
from simulate import RVContinuous, Simulation

# unpack command-line arguments
sample_size = int(sys.argv[1])

# target probability distribution
def target_cdf(x):
    if x >= 2.0 and x <= 3.0:
        return 0.25 * (x-2.0)**2
    elif x > 3.0 and x <= 6.0:
        return 0.25 + (x-3.0)*(9.0-x)/12.0
    else:
        return 0.0

# inverse of the target distribution
def inv_cdf(y):
    if y <= 0.25:
        return 2*(y**0.5 + 1)
    else:
        return 6 - 2*(3*(1-y))**0.5

# simulate and compare
rv = RVContinuous(support = [2.0, 6.0], cdf = target_cdf)
sim = Simulation(target_rv = rv, algorithm = 'inverse', inv_cdf = inv_cdf)
sim.generate(sample_size)
sim.compare(file_path = '../images/p2_{}.png'.format(sample_size))
