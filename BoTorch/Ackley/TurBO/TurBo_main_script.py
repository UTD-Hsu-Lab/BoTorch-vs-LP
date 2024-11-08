import os
import math
import warnings
from dataclasses import dataclass
import numpy as np
from scipy.stats import qmc

import torch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement, qUpperConfidenceBound
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

import subprocess
import record_data_log_file
import Norm_Dnorm
import heat_maps
import TestFunction
import turbo_state

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")


fun = Ackley(dim=6, negate=True).to(dtype=dtype, device=device)
fun.bounds[0, :].fill_(-32.768)
fun.bounds[1, :].fill_(32.768)
dim = fun.dim
lb, ub = fun.bounds


batch_size = 4
n_init = 24 
max_cholesky_size = float("inf")  # Always use Cholesky


n_LHS=99
output=[]
output1=[]
iteration=[]
noise = 0.00
ymin=-22.3

acquisition_function="ucb"
beta = 2



@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )



state = TurboState(dim=dim, batch_size=batch_size)

for run in range(n_LHS):
    output.append([]) 
    output1.append([])

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]), seed=run)
    
    X_turbo = turbo_state.get_initial_points(dim, n_init,run)
    Y_turbo = torch.tensor(
        [TestFunction.eval_objective(x,dim,fun) for x in X_turbo], dtype=dtype, device=device
    ).unsqueeze(-1)


    Y_turbo = Y_turbo + torch.randn_like(Y_turbo) * noise        
    max_y,idx = torch.max(Y_turbo,0) 
    max_x = X_turbo[idx,:]  

    # saving data
    output[run-1].append((max_x, max_y)) 
    output1[run-1].append((max_x, max_y))




    state = TurboState(dim, batch_size=batch_size, best_value=max(Y_turbo).item())

    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 512 if not SMOKE_TEST else 4
    N_CANDIDATES = min(5000, max(2000, 200 * dim)) if not SMOKE_TEST else 4

    torch.manual_seed(run)

    iterations=0

    while not iterations==50:  # Run until TuRBO converges
        # Fit a GP model
        if state.restart_triggered==True:
            state.length=state.length_min
        train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(
                nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)
            )
        )
        model = SingleTaskGP(
            X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Do the fitting and acquisition function optimization inside the Cholesky context
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            # Fit the model
            fit_gpytorch_mll(mll)

            # Create a batch
            X_next = turbo_state.generate_batch(
                beta,
                sampler,
                state=state,
                model=model,
                X=X_turbo,
                Y=train_Y,
                batch_size=batch_size,
                acqf=acquisition_function,
                n_candidates=N_CANDIDATES,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                            )

        Y_next = torch.tensor(
            [TestFunction.eval_objective(x,dim,fun) for x in X_next], dtype=dtype, device=device
        ).unsqueeze(-1)

                   
        

        # Print current status
        # print(
        #     f"{len(X_turbo)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}"
        # )

        yopt,idx = torch.max(Y_turbo,0) # Overall best value of y found, and its index
        Xopt = X_turbo[idx,:]  # Overall best value of x-input
        Xopt=unnormalize(Xopt, fun.bounds)
        output[run-1].append((Xopt, yopt))
        
        # print(f"\nBest Objective: {yopt}")
        # print(f"\nBest Predictor: {Xopt}")

        Y_mean=torch.mean(Y_turbo)
        Y_std= torch.std(Y_turbo)
        model.eval()
        with torch.no_grad():
                predicted=(model(X_turbo).mean)         
        predicted_dstd=Norm_Dnorm.destandardizer(predicted,Y_mean,Y_std)        
        Y_predicted, idx= torch.max(predicted_dstd,0)
        #print(predicted_dstd)

        print(yopt,Y_predicted)

        X_predicted = X_turbo[idx,:]
        X_predicted=unnormalize(X_predicted, fun.bounds)
        output1[run-1].append((X_predicted, Y_predicted)); iterations+=1
        # print('No of RUN= ',run,'\t Iteration = ',iterations,'Predicted =  ',Y_predicted,'\t Evaluated Maximum = ',yopt)


        Y_next= Y_next + torch.randn_like(Y_next) * noise   

        # Update state
        state = turbo_state.update_state(state=state, Y_next=Y_next)

        # Append data
        X_turbo = torch.cat((X_turbo, X_next), dim=0)
        Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)

       

    iteration.append(iterations)
    X=unnormalize(X_turbo,fun.bounds).numpy()

    
    heat_maps.parity_plot(X,X_turbo,model,Y_mean,Y_std,ymin,run,noise,beta)
    #heat_maps.predicted_model_perLHS(X,model,Y_mean,Y_std,ymin,run,noise)

    file='RUN_'+str(run+1)
    f=open(file,'w')
    f.close()


noise=int(noise*100)
outputfile='output_'+'Ackley'+'_beta'+str(beta)+'_noise'+str(noise)+'.csv'
record_data_log_file.record_csv(output, outputfile)
outputfile='output1_'+'Ackley'+'_beta'+str(beta)+'_noise'+str(noise)+'.csv'
record_data_log_file.record_csv(output1, outputfile)

print('Max iteration out of total LHS = ',np.max(iteration))