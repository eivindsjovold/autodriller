import numpy as np
import random

def generate_range():
    ## Set ranges
    range_a1 = np.array([0.5,1.9])
    range_a2 = np.array([0.000001,0.0005])
    range_a3 = np.array([0.000001,0.0009])
    range_a4 = np.array([0.000001,0.0001])
    range_a5 = np.array([0.5,2])
    range_a6 = np.array([0.4,1])
    range_a7 = np.array([0.3,1.5])
    range_a8 = np.array([0.3,0.6])

    ##Draw from uniform distribution
    a1 = random.uniform(range_a1[0],range_a1[1])
    a2 = random.uniform(range_a2[0],range_a2[1])
    a3 = random.uniform(range_a3[0],range_a3[1])
    a4 = random.uniform(range_a4[0],range_a4[1])
    a5 = random.uniform(range_a5[0],range_a5[1])
    a6 = random.uniform(range_a6[0],range_a6[1])
    a7 = random.uniform(range_a7[0],range_a7[1])
    a8 = random.uniform(range_a8[0],range_a8[1])

    return a1,a2,a3,a4,a5,a6,a7,a8




