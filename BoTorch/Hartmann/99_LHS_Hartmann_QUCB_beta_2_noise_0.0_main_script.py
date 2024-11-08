#%% Set Up
import torch
from scipy.stats import qmc
from botorch.test_functions import Hartmann
#%% Bayesian Optimization Loop
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import standardize
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
#from gpytorch.constraints import GreaterThan
from botorch.acquisition import qUpperConfidenceBound
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
import numpy as np
import subprocess
import record_data_log_file
import Norm_Dnorm
import plot
import inspect

import heat_maps







output=[]
output1=[]
n_LHS=99
n_lhs_samples=24
batch_size = 4
iterations = 50
noise = 0.0  # Need this > 0, same reason as sigma0 in Matlab's fitrgp must be > 0
dim = 6
ymax= 3.32237
beta = 2

for run in range(1,n_LHS+1,1):
    output.append([]) 
    output1.append([])    
    neg_hartmann6 = Hartmann(dim=dim, negate=True)  # Negative of Hartmann 6D for maximizing
    bounds = torch.tensor([[0.0] * 6, [1.0] * 6], dtype=float)  # Define x-domain
    opt_y = torch.tensor([3.32237]) # True max of y is 3.32237
    opt_x = torch.tensor([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.65730])
        # x-input that gives opt_y

    #%% Generate Initial Set of Data using Sobol sampling
    #train_x_sobol = draw_sobol_samples(bounds=bounds, n=n_lhs_samples, q=1).squeeze(1)
    
     # LHS
    sampler = qmc.LatinHypercube(d=dim, seed=run)
    train_x_lhs = sampler.random(n=n_lhs_samples)
    train_x=torch.tensor(train_x_lhs, dtype=float)
    

        # n=14 initial x-inputs;  .squeeze(1) converts 14x1x6 to 14x6 tensor
    train_y_true = neg_hartmann6(train_x).unsqueeze(-1)
        # Noiseless y-data;  .unsqueeze(-1) converts 1x14 to 14x1 tensor
    train_y = train_y_true + torch.randn_like(train_y_true) * noise        
    max_y,idx = torch.max(train_y,0) 
    max_x = train_x[idx,:]  

    # saving data
    output[run-1].append((max_x, max_y)) 
    output1[run-1].append((max_x, max_y))

     # Generates standardized y-values
    train_y_standard = standardize(train_y)
    train_yvar = torch.full_like(train_y, noise ** 2) # Variance of y-values
    outcome_transform = Standardize(m=1) # Unstandardizes GPR modeled y-values

    # Generate initial GPR model from training data:
    model = SingleTaskGP(train_x, train_y_standard, 
                        train_yvar, outcome_transform=outcome_transform)
    # Optimize hyperparameters by maximizing marginal log likelihood (same as Matlab's fitrgp):
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll);

    # Define Monte Carlo sampler for acquisition function & optimization
    # torch.Size([512]) means 512-by-1 column vector
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]), seed=run)

    Y_mean=torch.mean(train_y)
    Y_std= torch.std(train_y)
    model.eval()
    with torch.no_grad():
            predicted=(model(train_x).mean) 
    
    predicted_dstd=Norm_Dnorm.destandardizer(predicted,Y_mean,Y_std)
    # predicted=(model(train_x).mean) 
    # source_file = inspect.getfile(model().mean)
    # print(source_file)  
     
    Y_predicted= torch.max(predicted_dstd)

    print('Predicted =  ',Y_predicted)
        
    
    for j in range(1, iterations+1):
    # Choose acquisition function by *commenting in* one of following 3 lines:   
        AcqFn = qUpperConfidenceBound(model=model,beta=beta,sampler=sampler)
        #AcqFn = qLogExpectedImprovement(model=model,best_f=torch.max(max_y),sampler=sampler)
        #AcqFn = qLogNoisyExpectedImprovement(model=model,X_baseline=train_x,sampler=sampler)
        
        torch.manual_seed(seed=run)  # Use same random no. seed to keep restart conditions same
        
        # Generate next batch of x-inputs by jointly optimizing AcqFn:
        next_x, _ = optimize_acqf(acq_function = AcqFn, bounds = bounds,
                                q = batch_size, num_restarts = 20,
                                raw_samples = 100, 
                                options = {})
        
        next_y_true = neg_hartmann6(next_x).unsqueeze(-1)            
        next_y = next_y_true + torch.randn_like(next_y_true) * noise            
        next_ymax,idx = torch.max(next_y,0) # Best value of next_y, and its tensor index
        next_xmax = next_x[idx,:]  # Best value of next_x

        
        
        # Update data points
        train_x = torch.cat([train_x, next_x])
        train_y = torch.cat([train_y, next_y])
        train_y_standard = standardize(train_y)
        max_y = torch.cat([max_y, next_ymax]) # Running list of best y-values in each batch
        max_x = torch.cat([max_x, next_xmax])
        
        train_yvar = torch.full_like(train_y, noise ** 2)
        
        # Update GPR model with updated training data
        model = SingleTaskGP(train_x, train_y_standard, 
                            train_yvar, outcome_transform=outcome_transform)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll);
        
        # print(f"\nIteration {j} Next Batch Selection:")
        # print('Max y = ,',max(train_y))

        yopt,idx = torch.max(train_y,0) # Overall best value of y found, and its index
        Xopt = train_x[idx,:]  # Overall best value of x-input
        output[run-1].append((Xopt, yopt))
        
        # print(f"\nBest Objective: {yopt}")
        # print(f"\nBest Predictor: {Xopt}")

        Y_mean=torch.mean(train_y)
        Y_std= torch.std(train_y)
        model.eval()
        with torch.no_grad():
                predicted=(model(train_x).mean)         
        predicted_dstd=Norm_Dnorm.destandardizer(predicted,Y_mean,Y_std)        
        Y_predicted, idx= torch.max(predicted_dstd,0)

        X_predicted = train_x[idx,:]
        output1[run-1].append((X_predicted, Y_predicted))
        print('No of RUN= ',run,'\t Iteration = ',j,'Predicted =  ',Y_predicted,'\t Evaluated Maximum = ',yopt)
        
    
    # X=train_x.numpy()
    # heat_maps.parity_plot(X,train_x,model,Y_mean,Y_std,run,beta)
    # heat_maps.predicted_model_perLHS(X,model,Y_mean,Y_std,ymax,run,beta)

        
noise=int(noise*100)
outputfile='output_'+'Hartmann_QUCB'+'_beta'+str(beta)+'_noise'+str(noise)+'.csv'
record_data_log_file.record_csv(output, outputfile)
outputfile='output1_'+'Hartmann_QUCB'+'_beta'+str(beta)+'_noise'+str(noise)+'.csv'
record_data_log_file.record_csv(output1, outputfile)

plot.Learning_curve()

