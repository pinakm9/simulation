# This script requires 1 command line argument: sample size
import sys
from simulate import InverseComposition, RVContinuous

# unpack command-line arguments
sample_size = int(sys.argv[1])

# target probability distribution
def target_cdf(x, *args):
    return (x + x**3 + x**5)/3.0
