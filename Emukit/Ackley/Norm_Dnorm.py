import subprocess
import numpy as np
import time
import math
from matplotlib import pyplot as plt
import GPy
import csv


def normalizer(y, ymax, ymin):
    y_nor = (y-ymin)/(ymax-ymin)
    return y_nor

def denormalizer(y, ymax, ymin):
    y_dn = y * (ymax - ymin) + ymin
    return y_dn