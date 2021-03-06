# This script requires 1 command line argument: sample size
import sys
from simulate import RVContinuous, Simulation
from scipy.stats import binom, gamma
import numpy as np

# unpack command-line arguments
sample_size = int(sys.argv[1])

# raise all floating point errors
np.seterr(all = 'raise')

# problem parameters
n = 1000 # number of pocilyholders
p = 0.05 # probability of presenting a claim
mu = 800.0 # expected amount of claim
A = 50000.0 # least total amount according to the problem
epsilon = 1e-32 # a small number to weed out near-zero probabilties

# cdf for gamma distribution
def gamma_cdf(x, shape, scale):
    return gamma.cdf(x, shape, scale = scale)

# target probability distribution
def target_cdf(x, shapes, scale, weights):
    return sum((weights[i]*gamma_cdf(x, shape, scale = scale) for i, shape in enumerate(shapes)))

# mean finder for our random variable Y
def find_mean(shapes, scale, weights):
    return sum([weights[i]*shape for i, shape in enumerate(shapes)])*scale

# variance finder for our random variable Y
def find_var(shapes, scale, weights):
    return sum([weights[i]*shape for i, shape in enumerate(shapes)])*scale**2

# remove components with near zero probabilties for faster computation
significant_probabilites = []
significant_indices = []
for k in range(1, n+1):
    try:
        probability = binom.pmf(k, n, p)
    except:
        probability = 0.0
    if probability > epsilon:
        significant_indices.append(k)
        significant_probabilites.append(probability)

# print('number of significant components = {}'.format(len(significant_probabilites)))

# simulate and compare
sim_components = []
for i in significant_indices:
    gamma_rv = RVContinuous(cdf = gamma_cdf, shape = i+1, scale = mu)
    sim_components.append(Simulation(target_rv = gamma_rv, algorithm = 'gamma'))
rv =  RVContinuous(support = [0.0, np.inf], cdf = target_cdf, find_mean = find_mean, find_var = find_var,\
                   shapes = significant_indices, scale = mu, weights = significant_probabilites)
sim = Simulation(target_rv = rv, algorithm = 'composition', sim_components = sim_components, probabilties = significant_probabilites)
sim.generate(sample_size)
sim.compare(file_path = '../images/p10_{}.png'.format(sample_size))

# display results
print('simulated probability = {}\nactual probability = {}'\
      .format(1.0 - sim.ecdf(A), 1.0 - target_cdf(A, shapes = significant_indices, scale = mu, weights = significant_probabilites)))
