import subprocess
import numpy as np
import time
import math

from matplotlib import pyplot as plt
import GPy
import csv
from GPy.models import GPRegression


'''
Evaluates denormalized Ackley function for given coordinate(s)
x: list of coordinates, or numpy.array of lists of coordinates
a, b, c: Ackley function parameters
returns: numpy.ndarray of Ackley y-value(s) (denormalized)
'''
def ackley(x, a, b, c):
    if isinstance(x, list):
      x = np.array([x])
    d = x.shape[1]
    center = np.array([0]*d)
    # Evaluate Ackley function at the scaled LHS samples
    scaled_part_1 = -b*np.sqrt(1/d*np.sum((x-center)**2,axis=1))
    scaled_part_2 = 1/d*(np.sum(np.cos(c*(x-center)),axis=1))
    scaled_value = -(-a * np.exp(scaled_part_1) - np.exp(scaled_part_2) + a + np.exp(1))

    return scaled_value


# In[3]:


'''
Evaluates 4D or 6D Hartmann function for given coordinates (usually evaluated from 0 to 1 in every dimension)
x: numpy.array of lists of coordinates
'''
def hartmann(x):
    if isinstance(x, list):
      x = np.array([x])
    nvar = x.shape[1]
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 10**(-4) * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                             [2329, 4135, 8307, 3736, 1004, 9991],
                             [2348, 1451, 3522, 2883, 3047, 6650],
                             [4047, 8828, 8732, 5743, 1091, 381]])

    y = np.zeros(x.shape[0])
    for ii in range(4):
        inner = np.zeros(x.shape[0])
        for jj in range(nvar):
            xj = x[:, jj]
            Aij = A[ii, jj]
            Pij = P[ii, jj]
            inner += Aij * (xj - Pij) ** 2
        new = alpha[ii] * np.exp(-inner)
        #new = alpha[ii]*np.exp(-np.dot(x[:,:],A[ii,:]*(xj[ii,:]-Pij[ii,:])**2))
        y += new
    #y=(2.58+y)/1.94
    if(nvar == 4):
      y = -(1.1 - y) / 0.839
    
    return y

def global_hartmann_6D(x):
    nvar=6
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 10**(-4) * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                             [2329, 4135, 8307, 3736, 1004, 9991],
                             [2348, 1451, 3522, 2883, 3047, 6650],
                             [4047, 8828, 8732, 5743, 1091, 381]])

    y = 0
    for ii in range(4): 
        inner=0       
        for jj in range(nvar):
            xj = x[jj]
            Aij = A[ii, jj]
            Pij = P[ii, jj]
            inner += Aij * (xj - Pij) ** 2
        new = alpha[ii] * np.exp(-inner)
        y += new    
    return y


def global_ackley_6D():
    # Create a meshgrid for the x and y values
  def ackley_6d(x, y):
    return -20 * np.exp(-0.2 * np.sqrt((x**2 + y**2) / 2)) - np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) / 2) + 20 + np.e
  x = np.linspace(-32.768, 32.768, 400)
  y = np.linspace(-32.768, 32.768, 400)
  X, Y = np.meshgrid(x, y)
  Z = ackley_6d(X, Y)

  min_z = np.min(Z)
  max_z=np.max(Z)

  return min_z