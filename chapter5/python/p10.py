# This script requires 1 command line argument: sample size
import sys
from simulate import RVContinuous, Simulation
from scipy.stats import binom, gamma
import numpy as np

# unpack command-line arguments
sample_size = int(sys.argv[1])
np.seterr(all='raise')
# problem parameters
n = 1000 # number of pocilyholders
p = 0.05 # probabilty of presenting a claim
mu = 1.0 # expected amount of claim
A = 100.0 # least total amount according to the problem
epsilon = 1e-20 # a small number to weed out near-zero probabilties

# gamma distribution
def gamma_cdf(x, shape, scale):
    prod = 1.0
    total = 1.0
    multiplier = x*scale
    for i in range(1, int(shape)):
        try:
            prod *= multiplier/i
            total += prod
        except:
            return 1.0
    try:
        return 1.0 - total*np.exp(-multiplier)
    except:
        return 1.0

def gamma_pdf(x, shape, scale):
    prod = shape
    for i in range(1, shape):
        prod *= x*i/scale
        if prod < epsilon:
            return 0.0
    try:
        return prod*np.exp(-x/scale)/scale
    except:
        return 0.0

# remove components with near zero probabilties for faster computation
significant_probabilites = []
significant_indices = []
for k in range(n):
    try:
        probability = binom.pmf(k+1, n, p)
    except:
        probability = 0.0
    if probability > epsilon:
        significant_indices.append(k+1)
        significant_probabilites.append(probability)

#print('length of ... = {} ----> {}'.format(len(significant_probabilites), binom.pmf(1000,1000,0.05)))

# target probability distribution
def target_cdf(x):
    return sum((significant_probabilites[i]*gamma_cdf(x, k, mu) for i, k in enumerate(significant_indices)))

# target probability density
def target_pdf(x):
    return sum((significant_probabilites[i]*gamma_pdf(x, k, mu) for i, k in enumerate(significant_indices)))

# mean finder for gamma random variable
def find_mean(shape, scale):
    return shape*scale

# variance finder for gamma random variable
def find_var(shape, scale):
    return shape*scale**2

# simulate and compare
sim_components = []
for i in significant_indices:
    gamma_rv = RVContinuous(cdf = gamma_cdf, find_mean = find_mean, find_var = find_var, shape = i+1, scale = mu)
    sim_components.append(Simulation(target_rv = gamma_rv, algorithm = 'gamma'))
rv =  RVContinuous(support = [0.0, 500], cdf = target_cdf, pdf = target_pdf)
sim = Simulation(target_rv = rv, algorithm = 'composition', sim_components = sim_components, probabilties = significant_probabilites)
sim.generate(sample_size)
sim.compare(file_path = '../images/p10_{}.png'.format(sample_size))

print(1 - sim.ecdf(A), 1-target_cdf(A))
