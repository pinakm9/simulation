import numpy as np
from simulate import RVContinuous, Simulation

# a useful constant
c = 1.0-np.exp(-0.05)

# target probability distribution
def target_cdf(x):
    return (1.0 - np.exp(-x))/c

# inverse of the target distribution
def inv_cdf(y):
    return -np.log(1.0-c*y)

# simulate and compare
rv = RVContinuous(support = [0.0, 0.05], cdf = target_cdf)
sim = Simulation(target_rv = rv, algorithm = 'inverse', inv_cdf = inv_cdf)
sim.generate(1000)
sim.compare(file_path = '../images/p6.png')
print('simulated mean = {:.4f}\nexact mean = {:.4f}'.format(sim.mean, sim.rv.mean))
