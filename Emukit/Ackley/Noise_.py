import subprocess
import numpy as np
import time
import math
from matplotlib import pyplot as plt
import GPy
import csv


def noise_measure(x, std_noise):
    #Find the ground truth, and add Gaussian noise (defined by mean and std)
    np.random.seed()
    y = x + np.random.normal(loc= 0, scale = std_noise, size = x.shape)
    
    return y;