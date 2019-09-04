# This script requires 1 command line argument : sample size
import sys
from simulate import RVContinuous, Simulation

# unpack command-line arguments
sample_size = int(sys.argv[1])

# target probability distribution
def target_cdf(x):
    return (x + x**3 + x**5)/3.0

# cdf for the distribution x^n
def cdf(y, n):
    return x**n
# inverse for the distribution x^n
def inv_cdf(y, n):
    return y**(1.0/n)

# simulate and compare
rv = RVContinuous(support = [0.0, 1.0], cdf = target_cdf)
sim_components = [Simulation(target_rv = RVContinuous(support = [0.0, 1.0], cdf = cdf, n = 2*i+1), algorithm = 'inverse', inv_cdf = inv_cdf) for i in range(3)]
sim = Simulation(target_rv = rv, algorithm = 'composition', sim_components = sim_components, probabilties = [1.0/3.0]*3)
sim.generate(sample_size)
sim.compare(file_path = '../images/p8a_{}.png'.format(sample_size))
