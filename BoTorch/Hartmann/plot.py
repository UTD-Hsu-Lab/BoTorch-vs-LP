
import numpy as np
import time
import math    
from matplotlib import pyplot as plt
import GPy
import csv    
import subprocess    
import record_data_log_file
import Box_plot_Hartmann
import os
import re


def Learning_curve():
    n_lhs_samples = 10
    bs = 4
    total = 90    
    ground_y_max = 3.32237
    iterations = int((total-n_lhs_samples) / bs)
    start=0    
        

    beta = 2
    std_noise = 0

    directory_path=os.getcwd()
    pattern = "output_"
    matching_files = [filename for filename in os.listdir(directory_path) if filename.startswith(pattern)]
    if matching_files:
        filename = matching_files[0]
        print(filename)
        output = []
        path=os.getcwd()
        if os.path.exists(filename):
            with open(filename, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
                ctr = 0
                for row in reader:
                    if len(row) == 1:
                        ctr += 1
                        output.append([])
                    else:
                        output[ctr-1].append((row[1:], row[0]))
            
            #Box_plot_Hartmann.plotting(output,n_lhs_samples,total,bs,start,iterations,ground_y_max,beta,std_noise,filename)   
        os.chdir(path)
    

Learning_curve()
