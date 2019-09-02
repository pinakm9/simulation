import random, pathos
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from randomgen import RandomGenerator
import pathos.multiprocessing as mp
import multiprocess as mp_
from scipy.stats import rv_continuous
from statsmodels.distributions.empirical_distribution import ECDF
from utility import timer
import dill

def inverse_transform(random_variable):
    return random_variable.inv_dist(random.uniform(0.0, 1.0))


# A bare-bones base class for algorithms that simulate a random variable
class Simulation(object):
    def __init__(self):
        self.method = None # method of simulation/sampling
        self.random = random

    # generates samples using self.method
    @timer
    def generate(self, sample_size):
        self.size = sample_size # number of samples to be collected
        print(mp.cpu_count())
        pool = mp.Pool(mp.cpu_count())
        self.samples = pool.map(self.method, range(self.size)) # container for the collected samples
        #print(self.samples) --> debug
        pool.close()
        pool.join()
    @timer
    def generate_(self, sample_size):
        self.size = sample_size # number of samples to be collected
        self.samples = map(self.method, range(self.size)) # container for the collected samples


# Implements the inverse transform algorithm
class InverseTransform(Simulation):
    def __init__(self, inv_dist):
        Simulation.__init__(self)
        self.method = lambda *args: inv_dist(self.random.uniform(0.0, 1.0))

class AcceptReject(Simulation):
    def __init__(self, random_variable):
        pass

class RV():
    def __init__(self):
        self.random = random
    def f(self, *args):
        return self.random.uniform(0,1)

class RV_():
    def f(self, *args):
        return random.uniform(0,1)

if __name__ == "__main__":
        rv = InverseTransform(lambda y: y**0.5)
        n = int(1e6)
        rv.generate(n)
        rv.generate_(n)
