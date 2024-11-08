import numpy as np
import time
import math
from scipy.stats import qmc
from emukit.core import ParameterSpace, ContinuousParameter
from matplotlib import pyplot as plt
import GPy
import csv
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.bayesian_optimization.acquisitions import NegativeLowerConfidenceBound, ExpectedImprovement
from emukit.core.initial_designs.random_design import RandomDesign
import subprocess
import TestFunction
import Norm_Dnorm


import record_data_log_file

import os


def euclidean_distance_hartmann(x,iteration_num):   
    x0 = ((x[0]) - 0.20169)**2
    x1 = ((x[1]) - 0.150011)**2
    x2 = ((x[2]) - 0.476874)**2
    x3 = ((x[3]) - 0.275332)**2
    x4 = ((x[4]) - 0.311652)**2
    x5 = ((x[5]) - 0.6573)**2
    #print(np.sqrt(x0+x1+x2+x3+x4+x5))
    return np.sqrt(x0+x1+x2+x3+x4+x5)

def second_maxima_locator(Output,n_lhs_samples,total,bs,start,iterations,ground_y_max,beta,std_noise,filename):
    # Initialize iteration count as a global variable  
    output = []
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        ctr = 0
        for row in reader:
            
            if len(row) == 1: 
               ctr += 1
               output.append([]) 
            else:
               output[ctr-1].append((row[0], row[1:]))
    total_iter = len(output[0])
    data_X_up=[]
    data_X_down=[]
    

    ctr=0;
    ctr_1st_maxima=0;ctr_2nd_maxima=0

    first_max_index=[]
    second_max_index=[]
    for run in output:
        data_X = run[total_iter-1][1] 
               
        data=euclidean_distance_hartmann(data_X,iteration_num=49)     
        if data>0.2:
            
            ctr_2nd_maxima+=1            
        else:
            ctr_1st_maxima+=1
        
    f=open('LHS_percentage.xls','a')

    f.write(str(ctr_1st_maxima));f.write('\t')
    f.write(str(ctr_2nd_maxima));f.write('\n')
    f.close()
    print('First and Second counter by manual = ',ctr_1st_maxima,ctr_2nd_maxima,'\n\n')





