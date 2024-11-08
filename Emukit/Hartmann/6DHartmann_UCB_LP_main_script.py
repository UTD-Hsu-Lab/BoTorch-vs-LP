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
import Noise_
import record_data_log_file
import learning_curve_plotting_Hartmann
import Final_Model
import os

np.seterr(divide = 'ignore')

def loop(run, seed, nvar, r, n_samples, bs, total, std_noise, beta):
  #initialization
  output.append([]) # to keep track of output , for utility function u(D)= Max(y)
  output1.append([]) # to keep track of output , for utility function u(D)= Max(mu_D(X))
  kernel_paramters.append([])
  kernel_paramters_lengthscale.append([]) 
  model.append([])
  model_data_X.append([])
  model_data_Y.append([])
  
  
  # LHS defining
  sampler = qmc.LatinHypercube(d=nvar, seed=seed)
  x_nor = sampler.random(n=n_samples)

  # Scale samples & evaluate function
  xmax = np.array([r]*nvar)
  xmin = np.array([0]*nvar)
  scaled_lhs_samples = Norm_Dnorm.denormalizer(x_nor, xmax, xmin)
  y_lhs_samples = TestFunction.hartmann(scaled_lhs_samples)
  
  #parameter space defining
  parameter_list = []
  for i in range(nvar):
    parameter_list.append(ContinuousParameter('x' + str(i+1), 0, 1))
  parameter_space = ParameterSpace(parameter_list)

  # evaluating Hartmann function ground truth range, 
  global ymax, ymin
  if run == 0:
    X_=([0.20169,0.150011,0.476874,0.275332,0.311652,0.6573])
    ymax=TestFunction.global_hartmann_6D(X_)       
    ymin = 0
  
  #Adding noise
  y_nor = Noise_.noise_measure(Norm_Dnorm.normalizer(y_lhs_samples, ymax, ymin), std_noise)
  Y_out = Norm_Dnorm.denormalizer(y_nor, ymax, ymin)
  y_nor = y_nor.reshape(-1,1)

  # Adding LHS to output array
  output[run].append((scaled_lhs_samples[np.argmax(Norm_Dnorm.denormalizer(y_nor, ymax, ymin))], np.max(Norm_Dnorm.denormalizer(y_nor, ymax, ymin))))
  output1[run].append((scaled_lhs_samples[np.argmax(Norm_Dnorm.denormalizer(y_nor, ymax, ymin))], np.max(Norm_Dnorm.denormalizer(y_nor, ymax, ymin))))
    
    
    # Loop setup
  current_iter = 0  
  Y_new_n = [] # for first run of the loop below
  for current_iter in range(iterations):
    #input and output defining    
    X, Y = [x_nor, y_nor]
    #GP model set up
    ker = GPy.kern.Matern52(input_dim = nvar, ARD =True)
    ker.lengthscale.constrain_bounded(1e-3, 1, warning=False)
    ker.variance.constrain_bounded(1e-3, 1, warning=False)

    #model defining
    model_gpy = GPRegression(X , -Y, ker)   #Emukit is a minimization tool; need to make Y negative
    model_gpy.randomize()
    model_gpy.optimize_restarts(num_restarts=10,verbose =False, messages=False)
    objective_model = GPyModelWrapper(model_gpy)
    f_obj =  objective_model.model.predict

    #Acquisition Function Defining
    # Upper Confidence Bound (UCB)    
    acquisition = NegativeLowerConfidenceBound(objective_model, beta = beta)
    # Expeceted Improvement (EI)
    #acquisition = ExpectedImprovement(objective_model, jitter=jitter)
    
    
    # collecting BO points
    bayesopt= BayesianOptimizationLoop(model=objective_model,
                                      space=parameter_space,
                                      acquisition=acquisition,
                                      batch_size = bs) #batchsize may need to be >bs due to duplication  
    X_new = bayesopt.candidate_point_calculator.compute_next_points(bayesopt.loop_state.X)


    # evaluation of model predicted objective value and coordinate    
    predicted, uncertainity=(model_gpy.predict(X))
    predicted=(Norm_Dnorm.denormalizer(predicted, ymax, ymin))
    max_y_coordinate=np.argmin(model_gpy.predict(X))    
    Y_predicted= -np.min(predicted)
    max_y = -np.min(np.transpose(bayesopt.loop_state.Y))
    max_y_coor = np.argmin(np.transpose(bayesopt.loop_state.Y))
    

    #Printing
    print(
        "                                            curent iteration =",
        current_iter,'\t\t Current Loop',run+1,'\nY_coordinate_predicted = ',max_y_coordinate,'\t and predicted  = ',Y_predicted,'\n')     


    
    # Evaluation of ground truth value and adding noise
    X_new_dn = Norm_Dnorm.denormalizer(X_new,xmax,xmin)   
    Y_new_n = Noise_.noise_measure(Norm_Dnorm.normalizer(TestFunction.hartmann(X_new_dn),ymax,ymin), std_noise)   


    # updating the model    
    Y_new_n=Y_new_n.reshape(-1,1)
    y_nor = np.append(y_nor, Y_new_n, axis=0)
    x_nor = np.append(x_nor, X_new, axis=0)  


    # saving output 
    output[run].append((Norm_Dnorm.denormalizer(bayesopt.loop_state.X[max_y_coor], r, -r), Norm_Dnorm.denormalizer(max_y, ymax, ymin)))
    output1[run].append((Norm_Dnorm.denormalizer(bayesopt.loop_state.X[max_y_coordinate], r, -r), Y_predicted))
    model[run].append(f_obj)
    gaussian_variance=(model_gpy.Gaussian_noise.variance[0])    
    kernel_paramters[run].append((ker.variance[0],gaussian_variance))  
    kernel_paramters_lengthscale[run].append(ker.lengthscale[0:6])  

    #updaing iteration     
    current_iter += 1
 

  
  # plotting 3D heat maps and parity plot after each run
  X=np.array(bayesopt.loop_state.X)
  Y=(Norm_Dnorm.denormalizer(np.array(bayesopt.loop_state.Y),ymax,ymin))  
  percentile_no=run;final_model=model[run];
  Final_Model.plot_contour_per_LHS(nvar,f_obj,ymax,ymin,xmax,xmin,X,Y,std_noise,beta,percentile_no,final_model)

  
#initial set_up
seed_0 = 0
nvar = 6
r = 1
n_lhs_samples = 24
bs = 4
total = 228 
std_noise = 0.0
beta = 1
iterations = int((total-n_lhs_samples) / bs) ;
n_runs = 2

# data_record variable defined
output = []
output1 =[]
noise_variance=[]
kernel_paramters=[]
kernel_paramters_lengthscale=[]
model=[]
model_data=[]
model_data_X=[]
model_data_Y=[]


#loop started from here
for i in range(n_runs):
  loop(i, seed=seed_0+i, nvar=nvar, r=r, n_samples=n_lhs_samples, bs=bs, total=total, std_noise=std_noise, beta=beta)
  

#Data Recording and plotting
noise=int(std_noise*100)
outputfile='output_max(y)'+'Hartmann_LP'+'beta'+str(beta)+'_noise'+str(noise)+'.csv'
outputfile1='output_max(mu_D(X))'+'Hartmann_LP'+'beta'+str(beta)+'_noise'+str(noise)+'.csv'
record_data_log_file.record_csv(output, outputfile)
record_data_log_file.record_csv(output1, outputfile1)

#kernel parameters data recording and plotting
file='kernel_parameters_'+'beta'+str(beta)+'_noise'+str(noise)+'.csv';ker_variance_filename=file
record_data_log_file.record_csv_variance(kernel_paramters, file)

file='kernel_parameters_lengthscale_'+'beta'+str(beta)+'_noise'+str(noise)+'.csv';ker_lengthscale_filename=file
record_data_log_file.record_csv_lengthscale(kernel_paramters_lengthscale, file)


#Learning curve plotting 

learning_curve_plotting_Hartmann.plotting(output1,n_lhs_samples,total,bs,iterations,ymax,beta,std_noise,outputfile1,ker_variance_filename,ker_lengthscale_filename)









