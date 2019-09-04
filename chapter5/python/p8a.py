# This script requires 1 command line argument : sample size
import sys
from simulate import  Composition, InverseTransform, RVContinuous

# unpack command-line arguments
sample_size = int(sys.argv[1])

# target probability distribution
def target_cdf(x, *args):
    return (x + x**3 + x**5)/3.0

# cdf for the distribution x^n
def cdf(y, n):
    return x**n
# inverse for the distribution x^n
def inv_cdf(y, n):
    return y**(1.0/n)

# simulate and compare
sim_components = [InverseTransform(RVContinuous(support = [0.0, 1.0], cdf = cdf, inv_cdf = inv_cdf, params = n)) for n in range(1,6,2)]
sim = Composition(RVContinuous(support = [0.0, 1.0], cdf = target_cdf), sim_components, [1.0/3.0]*3)
sim.generate(sample_size)
sim.compare(file_path = '../images/p8a_{}.png'.format(sample_size))
