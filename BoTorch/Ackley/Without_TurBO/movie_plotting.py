import subprocess
import Norm_Dnorm
import TestFunction
import numpy as np
import time
import math
from matplotlib import pyplot as plt
import GPy
import csv
from emukit.core.initial_designs.random_design import RandomDesign
import os
import re
import pickle
from emukit.core import ParameterSpace, ContinuousParameter
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import record_data_log_file
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap

import torch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


def saveing_in_folder(folder_name,plot_filename):
    folder_path = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    plot_filepath = os.path.join(folder_path, plot_filename)
    plt.savefig(plot_filepath)

    # Close the plot (optional)
    #plt.close()

def squeeze(a):
    if len(a.shape) > 2:
        siz = a.shape
        siz = tuple(dim for dim in siz if dim != 1)
        if len(siz) == 1:
            b = a.reshape(siz[0], 1)
        else:
            b = a.reshape(siz)
    else:
        b = a
    return b
def meshgrid():
    n= 11
    Xdomain = np.array([[0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1]])    

    x1 = np.linspace(Xdomain[0, 0], Xdomain[0, 1], n)
    x2 = np.linspace(Xdomain[1, 0], Xdomain[1, 1], n)
    x3 = np.linspace(Xdomain[2, 0], Xdomain[2, 1], n)
    x4 = np.linspace(Xdomain[3, 0], Xdomain[3, 1], n)
    x5 = np.linspace(Xdomain[4, 0], Xdomain[4, 1], n)
    x6 = np.linspace(Xdomain[5, 0], Xdomain[5, 1], n)

    # Convert the arrays to column vectors (similar to your original code)
    x1 = x1[:, np.newaxis]
    x2 = x2[:, np.newaxis]
    x3 = x3[:, np.newaxis]
    x4 = x4[:, np.newaxis]
    x5 = x5[:, np.newaxis]
    x6 = x6[:, np.newaxis]

    # Use np.meshgrid to create the grid of coordinates
    X1, X2, X3, X4, X5, X6 = np.meshgrid(x1, x2, x3, x4, x5, x6, indexing='ij')
    X = np.column_stack([X1.ravel(), X2.ravel(), X3.ravel(), X4.ravel(), X5.ravel(), X6.ravel()])

    return X,x1,x2,x3,x4,x5,x6,n
    

def model_prediction_per_iteration(model,ymax,ymin,xmax,xmin,frame,Y_mean,Y_std):
    X_,x1,x2,x3,x4,x5,x6,n=meshgrid()


    # Compute y_pred using the ackley6D function
    X_=torch.tensor(X_, dtype=float)
    f_obj=model[frame]
    predicted = f_obj(X_).mean
    
    y_pred=Y_pred=Norm_Dnorm.destandardizer(predicted,Y_mean,Y_std)
    y_pred=y_pred.detach().numpy();

    y_pred = np.array(y_pred)       
    X,x1,x2,x3,x4,x5,x6,n=meshgrid()
    X=np.array(X)

    idx = np.argmax(y_pred)
    X_=Norm_Dnorm.denormalizer(X_, xmax, xmin);
    X1max, X2max, X3max, X4max, X5max, X6max = X_[idx, 0], X_[idx, 1], X_[idx, 2], X_[idx, 3], X_[idx, 4], X_[idx, 5]
    X1true, X2true, X3true, X4true, X5true, X6true = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    Ycube = np.reshape(y_pred, (n, n, n, n, n, n))
    M12 = np.max(Ycube, axis=(2, 3, 4, 5));M12 = np.squeeze(M12)
    M13 = np.max(Ycube, axis=(1, 3, 4, 5));M13 = np.squeeze(M13)
    M14 = np.max(Ycube, axis=(1, 2, 4, 5));M14 = np.squeeze(M14)
    M15 = np.max(Ycube, axis=(1, 2, 3, 5));M15 = np.squeeze(M15)
    M16 = np.max(Ycube, axis=(1, 2, 3, 4));M16 = np.squeeze(M16)

    return M12,M13,M14,M15,M16,X1max,X2max,X3max,X4max,X5max,X6max,X1true,X2true,X3true,X4true,X5true,X6true;

def update(frame,X1,X2,f_obj,X,xmax,xmin,ymax,ymin,bound,frame_length,Y_mean,Y_std):       
                                #..........case-1............ 
        
    fig1 = plt.figure(1)
    fig1.set_size_inches(4, 3)
    ax = fig1.add_subplot(111, projection='3d',computed_zorder=False)

    #prediction per iteration
    M12,M13,M14,M15,M16,X1max,X2max,X3max,X4max,X5max,X6max,X1true,X2true,X3true,X4true,X5true,X6true=model_prediction_per_iteration(f_obj,ymax,ymin,xmax,xmin,frame,Y_mean,Y_std)
    index1=1;index2=2
    # colormap case 1
    colorbar_offset = [ymin, 0, 0]
    n_bins = 2585  # Number of bins in the colormap
    limit=-ymin/2585
    c_offset=colorbar_offset[0]
    #colors = ['#CBE7C9', '#14720E']



    # colormap case 1
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
    }

    # Define the positions for color transitions
    positions = [0.0, 0.20, 1.0]  # First 20% transitions quickly, remaining 80% transitions slowly

    # Define colors and their corresponding positions for transition
    start_color = (0.0588, 0.5529, 0.0314)  # RGB values of #26580F
    end_color = (0.937,0.984,0.169) # yellow

    fast_color = (start_color[0] * 0.2, start_color[1] * 0.2, start_color[2] * 0.2)  # 20% of start color
    colors = [fast_color, start_color, end_color]

    for pos, color in zip(positions, colors):
        r, g, b = color
        cdict['red'].append((pos, r, r))
        cdict['green'].append((pos, g, g))
        cdict['blue'].append((pos, b, b))

    custom_cmap = LinearSegmentedColormap('CustomColormap', cdict)
    cmap = custom_cmap 

    ax.plot_surface(X2, X1, M12,cmap='viridis')

    if bound==34:
        xyticks=[-30,0,30]  
        zticks=[-20,-10,0]  
        ax.set_zticks(zticks)   
    elif bound==2:
        xyticks=[-2,0,2]
    elif bound==5:
        xyticks=[-4,0,4]
    plt.xlabel('x'+str(index1));plt.xticks(xyticks)
    plt.ylabel('x'+str(index2));plt.yticks(xyticks)
    plt.tick_params(direction='in',pad=0.1) 

    # if index1==6:           
    #     cb = plt.colorbar(pad=0.05)
    #     ticks=[0.0,0.4,0.8,1.2,1.6,2.0,2.4,2.8,3.2]
    #     cb.set_ticks(ticks)                    
    #     cb.set_label('Max(μ_D($\mathbf{X}$))', labelpad=0) 

    # contour line
    n_contourline_bins=10
    limit=-ymin/n_contourline_bins    
    # colormap case 1..orange color
    if bound==2:
        plt.contour(X2, X1, M12, levels = np.arange(n_contourline_bins+1)*limit+c_offset, colors='#E3BB18', linestyles='dashed',linewidth=0.7)


    
    c_offset=colorbar_offset[1]
    limit=1
    # Create a segmented color map with varying segment lengths
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
    }
    positions = [0.0, 0.4, 0.7, 1.0]  # First 40% transitions quickly, remaining 60% transitions slowly
    # Define colors and their corresponding positions for transition
    # case...1.............................
    #colors = [(0.97, 1, 0.01), (1, 0.01, 0.01), (0.51, 0.01, 0.01)] # yellow to red
    colors = [(247/255, 216/255, 241/255), (251/255, 93/255, 88/255), (222/255, 3/255, 13/255), (89/255, 2/255, 6/255)]  #pink to# Gradient within red
    # case 2...................
    #colors = [(1, 1, 1), (1, 0, 0), (0.51, 0.08, 0.08)]  # White to red to #831414
    for pos, color in zip(positions, colors):
        r, g, b = color
        cdict['red'].append((pos, r, r))
        cdict['green'].append((pos, g, g))
        cdict['blue'].append((pos, b, b))

    custom_cmap = LinearSegmentedColormap('CustomRed', cdict)


    cmap=custom_cmap  
    normalize=plt.Normalize(0,frame_length)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    sm.set_array([])  # You can set this to a specific array if needed
    # cbar = plt.colorbar(sm, pad=0.01)
    # cbar.set_label('Iteration', labelpad=0) 
    # num_ticks = 2  # Adjust the number of ticks as needed
    # cbar.locator=MaxNLocator(integer=True, nbins=num_ticks)
    # cbar.update_ticks()
    plt.xlim(-bound,bound)
    plt.ylim(-bound,bound)

    Y_truth=TestFunction.ackley(X,a=20,b=0.2,c=2*np.pi)
    n_lhs=24;
    marker_size=20
    for i in range(n_lhs):
        x=X[i]
        y_truth=Y_truth[i]
        ax.scatter(x[index1-1],x[index2-1],y_truth,s=marker_size, zorder=2, color='royalblue',marker='o')
    ax.scatter(x[index1-1],x[index2-1],y_truth,s=marker_size, zorder=2, color='royalblue',marker='o')

    plt.title(f'Iteration {frame+1}/{frame_length}')
    index_start=24;index_end=int(24+(frame)*4)+4
    color=cmap(normalize(frame))

    ctr=0
    for i in range(index_start,index_end,1):
        temp=(i)%4
        if temp==0:
            index=int(float(i/4))
            color=cmap(normalize(index))
        x=X[i]  
        y=Y_truth[i]
        print(x[index1-1],x[index2-1])                     
        ax.scatter(x[index1-1],x[index2-1],y,s=marker_size, zorder=2, color=[color])
    ax.scatter(x[index1-1],x[index2-1],y,s=marker_size, zorder=2, color=[color],marker='o',label='BO Points')
    gt_marker_size=100
    height_for_visibility=0.0
    xtruez1=0+height_for_visibility; 
    ax.scatter(X1true, X2true,xtruez1, s=gt_marker_size, zorder=3, color='#12a6ce', marker='x', label='GT Max',alpha=1.0,depthshade=False)    
                      
    print('Frame No==',frame)
    # legend_labelsize=8
    # legend = plt.legend()
    # legend.get_frame().set_alpha(0.6)
    # plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1.2),fontsize=legend_labelsize)
    plt.tight_layout()
    plt.savefig(f'frame_{frame}_bound_{bound}.png')  # Save each frame as a PNG image
    plt.close()
           


def movie_plot(beta, x1, x2, BO_model,model_X, xmax, xmin, X1true, X2true,ymax,ymin,bound,Y_mean,Y_std):
    
    X1, X2 = np.meshgrid(x1, x2)    
    X1=Norm_Dnorm.denormalizer(X1, xmax, xmin);X2=Norm_Dnorm.denormalizer(X2, xmax, xmin);

    frame_length=int((len(model_X)-24)/4) 
    
    for frame_number in range(frame_length):
        temp=int((frame_number+1)%4)
        if frame_number<=20:
              # Create a new figure for each frame
            update(frame_number, X1, X2, BO_model, model_X, xmax, xmin, ymax, ymin, bound, frame_length,Y_mean,Y_std)
            
        elif frame_number>20 and temp==0:
            
            update(frame_number, X1, X2, BO_model, model_X, xmax, xmin, ymax, ymin, bound, frame_length,Y_mean,Y_std)
            



    directory_path=os.getcwd()

    file_list = os.listdir(directory_path)
    file_list = [file for file in file_list if re.match(r'frame_\d+\_bound_\d+\.png', file)]
    file_list = sorted(file_list, key=lambda x: int(re.search(r'(\d+)', x).group(1)))

    print(file_list)

    def animate(frame):
        plt.clf()
        img = plt.imread(os.path.join(directory_path, file_list[frame]))
        plt.imshow(img)
        plt.axis('off')
        #plt.title(f'Frame {frame}/{len(file_list)}')

    # Set up the figure and create the animation
    fig = plt.figure(figsize=(4,3))
    ani = animation.FuncAnimation(fig, animate, frames=len(file_list), interval=200)
    writer = animation.writers['pillow'](fps=1.25)
    animation_name='2D_plot_movie_'+'x_range_'+str(bound)+'.gif'
    ani.save(animation_name, writer=writer)
 

    
    
   



    
