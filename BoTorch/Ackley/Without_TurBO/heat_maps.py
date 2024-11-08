import subprocess
import Norm_Dnorm
import TestFunction
import movie_plotting
import numpy as np
import time
import math
from matplotlib import pyplot as plt
import GPy
import csv

import os
import re
import pickle

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import record_data_log_file
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap


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


def saveing_in_folder(folder_name,plot_filename):
    folder_path = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    plot_filepath = os.path.join(folder_path, plot_filename)
    plt.savefig(plot_filepath)

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

def parity_plot(X,train_x,model,Y_mean,Y_std,ymin,LHS_no,noise):
    
    
    Y_truth=TestFunction.ackley(X,a=20,b=0.2,c=2*np.pi);Y_truth=np.array(Y_truth).reshape(len(Y_truth),1)  
    predicted=(model(train_x).mean)
    Y_pred=Norm_Dnorm.destandardizer(predicted,Y_mean,Y_std)
    Y_pred=Y_pred.detach().numpy();Y_pred=np.array(Y_pred).reshape(len(Y_pred),1)  

    print('Y_truth =  ',Y_truth)

    print('Y_pred =  ',Y_pred)

    fig= plt.figure(figsize=(2.2,2))
    ax = plt.gca()
    bounds=-23,0
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)

    plt.scatter(Y_truth,Y_pred,s = 30, facecolors='gray', alpha = 0.5, edgecolor = 'blue')
    x = np.linspace(min(np.min(Y_truth), np.min(Y_pred)), max(np.max(Y_truth), np.max(Y_pred)), 100)
    plt.plot(x, x, 'r--')  # Red dashed line indicating y = x

    x=Y_truth.flatten();y=Y_pred.flatten()
    mean_abs_err = np.mean(np.abs(x-y))
    rmse = np.sqrt(np.mean((x-y)**2))
    rmse_std = rmse / np.std(y)
    z = np.polyfit(x,Y_pred.flatten(), 1)
    y_hat = np.poly1d(z)(x)
    text = f"$ RMSE = {rmse:0.6f}$"

    filename='parity_plot'+'Percentile_'+str(LHS_no)+'.xls'
    f=open(filename,'w')

    f.write(str(LHS_no));f.write('\t')
    f.write('UN RMSE = ->');f.write('\t')
    f.write(str(rmse));f.write('\n')
    f.write('N RMSE = ->');f.write('\t')
    f.write(str(rmse/np.abs(ymin)));f.write('\n')
    
    for index, (x, truth, pred) in enumerate(zip(X, Y_truth, Y_pred)):
        f.write(str(x[0]));f.write('\t')
        f.write(str(x[1]));f.write('\t')
        f.write(str(x[2]));f.write('\t')
        f.write(str(x[3]));f.write('\t')
        f.write(str(x[4]));f.write('\t')
        f.write(str(x[5]));f.write('\t')
        f.write(str(truth[0]));f.write('\t')
        f.write(str(pred[0]));f.write('\n')

    f.close()

    #plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,verticalalignment='top')
    plt.xlabel('Ground Truth')
    plt.ylabel('Model Prediction')
    ticks=[-20,-15,-10,-5,0]
    plt.xticks(ticks)
    plt.yticks(ticks)  
    plt.tick_params(direction='in') 

    plt.tight_layout()
    
    folder_name='parity_plot'   
    plot_filename='parity_plot_LHS_no'+str(LHS_no)+'.svg'
    saveing_in_folder(folder_name,plot_filename)
    plt.close()

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

def create_subplot(subplot_num, subplot_row,subplot_column, x1, x2, M, X,Xmax1,Xmax2, Xtrue1,Xtrue2,X1true1,X1true2,ymin,LHS_No,noise,bound,index1,index2):
    
    if subplot_column!=1 or subplot_row!=1:
        plt.subplot(subplot_row, subplot_column, subplot_num)
    else:
        fig1 = plt.figure(1)
        fig1.set_size_inches(2.2, 2)
    X1, X2 = np.meshgrid(x1, x2)
    xmax=32.768;xmin=-32.768;
    X1=Norm_Dnorm.denormalizer(X1, xmax, xmin);X2=Norm_Dnorm.denormalizer(X2, xmax, xmin);
    colorbar_offset = [ymin, 0, 0]
    n_bins = 2585  # Number of bins in the colormap
    limit=-ymin/n_bins
    c_offset=colorbar_offset[0]


    # colormap case 1
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
    }

    # Define the positions for color transitions
    positions = [0.0, 0.75, 1.0]  # First 20% transitions quickly, remaining 80% transitions slowly

    # Define colors and their corresponding positions for transition
    start_color = (0.0588, 0.5529, 0.0314)  # RGB values of #26580F
    end_color = (1.0, 1.0, 1.0)  # White

    fast_color = (start_color[0] * 0.2, start_color[1] * 0.2, start_color[2] * 0.2)  # 20% of start color
    colors = [fast_color, start_color, end_color]

    for pos, color in zip(positions, colors):
        r, g, b = color
        cdict['red'].append((pos, r, r))
        cdict['green'].append((pos, g, g))
        cdict['blue'].append((pos, b, b))

    custom_cmap = LinearSegmentedColormap('CustomColormap', cdict)
    cmap = custom_cmap 

    plt.contourf(X2, X1, M, 50, levels = np.arange(n_bins+1)*limit+c_offset, cmap=cmap)

    if bound==34:
        xyticks=[-30,-10,0,10,30]      
    elif bound==2:
        xyticks=[-2,-1,0,1,2]
    elif bound==5:
        xyticks=[-4,-2,0,2,4]
    plt.xlabel('x'+str(index1));plt.xticks(xyticks)
    plt.ylabel('x'+str(index2));plt.yticks(xyticks)
    plt.tick_params(direction='in')

    # if index1==6:           
    #     cb = plt.colorbar(pad=0.05)
    #     ticks=[0.0,0.4,0.8,1.2,1.6,2.0,2.4,2.8,3.2]
    #     cb.set_ticks(ticks)                    
    #     cb.set_label('Max(μ_D($\mathbf{X}$))', labelpad=0) 

    # contour line
    n_contourline_bins=10
    limit=-ymin/n_contourline_bins    
    # colormap case 1..orange color
    if index1!=6:
        plt.contour(X2, X1, M, levels = np.arange(n_contourline_bins+1)*limit+c_offset, colors='#E3BB18', linestyles='dashed',linewidth=0.7)



    ## BO points 

    c_offset=colorbar_offset[1]
    limit=1
    # Create a segmented color map with varying segment lengths
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
    }

    # Define the positions for color transitions
    positions = [0.0, 0.4, 0.7, 1.0]  # First 20% transitions quickly, remaining 60% transitions slowly
    colors = [(247/255, 216/255, 241/255), (251/255, 93/255, 88/255), (222/255, 3/255, 13/255), (89/255, 2/255, 6/255)]  #pink to# Gradient within red
            

    for pos, color in zip(positions, colors):
        r, g, b = color
        cdict['red'].append((pos, r, r))
        cdict['green'].append((pos, g, g))
        cdict['blue'].append((pos, b, b))

    custom_cmap = LinearSegmentedColormap('CustomRed', cdict); cmap=custom_cmap

    normalize = plt.Normalize(0, len(X))
    count=0
    length=len(X)
    count1=0;count2=0   

    normalize1=plt.Normalize(0,((len(X)-24)/4))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize1)
    sm.set_array([])  # You can set this to a specific array if needed
    # Add a color bar to the right of the figure
    if index1==7:
        cbar = plt.colorbar(sm, pad=0.05)#
        #cbar.set_label('Iteration', labelpad=0,fontsize=18)
        cbar.set_label('Iteration', labelpad=0)            
        num_ticks = 3  # Adjust the number of ticks as needed
        cbar.locator=MaxNLocator(integer=True, nbins=num_ticks)
        #cbar.ax.tick_params(labelsize=15)
        cbar.update_ticks()

    marker_size=20
    if index1<6:
        for x in X:
            #print(x)
            color=cmap(normalize(count));count1=0;                            
            if count==(len(X)-1):
                plt.scatter(x[index1-1],
                        x[index2-1],
                        s=marker_size, zorder=5, color=[color],marker='o',label='BO Points')
            else:  
                if count<24:
                    plt.scatter(x[index1-1],
                        x[index2-1],
                        s=marker_size, zorder=5, color='#25EF18')  #medium green
                    if count==23:
                        plt.scatter(x[index1-1],x[index2-1],s=20, zorder=5, color='#25EF18',label='LHS Points')  #medium green

                else:
                    plt.scatter(x[index1-1],
                            x[index2-1],
                            s=marker_size, zorder=5, color=[color])
            count+=1
    
    gt_marker_size=80
    plt.scatter(Xtrue1, Xtrue2, s=gt_marker_size, zorder=5, color='cyan', marker='x', label='GT Max')    
           
    # if index1==8:            
    #     legend = plt.legend()
    #     legend.get_frame().set_alpha(0.6)
    #     legend_font = { 'size': 18}
    #     #plt.legend(prop=legend_font,loc='upper left', bboXto_anchor=(1.2, 1.25))
    #     plt.legend(loc='upper left', bboXto_anchor=(1.2, 1.25))

    plt.xlim(-bound,bound)
    plt.ylim(-bound,bound)

    if subplot_column==1 and subplot_row==1:
        plt.tight_layout()
        folder_name='Heat_Maps'+'LHS_No'+str(LHS_No)+'_noise'+str(noise)
        plot_filename='X'+str(index1)+'X'+str(index2)+'bound_'+str(bound)+'.png'
        saveing_in_folder(folder_name,plot_filename)
        plt.close()

    



def create_subplot_surface(subplot_num, subplot_row,subplot_column, x1, x2, M, X,Xmax1,Xmax2, Xtrue1,Xtrue2,X1true1,X1true2,ymin,LHS_No,noise,bound,index1,index2):
    
    fig1 = plt.figure(figsize=(2.2,2))
    plt.style.use('seaborn-paper')
    #fig1.set_size_inches(2.2, 2)
    if bound==34:
        ax=fig1.add_subplot(1, 1, 1,projection='3d',computed_zorder=False)
    else:
        ax=fig1.add_subplot(1, 1, 1)
    

    X1, X2 = np.meshgrid(x1, x2)
    xmax=32.768;xmin=-32.768;
    X1=Norm_Dnorm.denormalizer(X1, xmax, xmin);X2=Norm_Dnorm.denormalizer(X2, xmax, xmin);
    
    colorbar_offset = [ymin, 0, 0]
    n_bins = 2585  # Number of bins in the colormap
    limit=-ymin/n_bins
    c_offset=colorbar_offset[0]

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
    end_color = (0.937,0.984,0.169) 

    fast_color = (start_color[0] * 0.2, start_color[1] * 0.2, start_color[2] * 0.2)  # 20% of start color
    colors = [fast_color, start_color, end_color]

    for pos, color in zip(positions, colors):
        r, g, b = color
        cdict['red'].append((pos, r, r))
        cdict['green'].append((pos, g, g))
        cdict['blue'].append((pos, b, b))

    custom_cmap = LinearSegmentedColormap('CustomColormap', cdict)
    cmap = custom_cmap 

    if bound==34:
        cbar= ax.plot_surface(X2, X1, M,cmap=cmap,zorder=1)
    else:
        plt.contourf(X2, X1, M, 50, levels = np.arange(n_bins+1)*limit+c_offset, cmap=cmap)
    
    zticks=[-20,-10,0]
    if bound==34:
        plt.xlabel('x'+str(index1),labelpad=0.1)
        plt.ylabel('x'+str(index2),labelpad=0.1)
        x_ticks=[-30,0,30]
        plt.xticks(x_ticks)
        plt.yticks(x_ticks)
        ax.set_zticks(zticks)            
        plt.tick_params(direction='in',pad=0.1)   

    elif bound==2:
        plt.xlabel('x'+str(index1))
        plt.ylabel('x'+str(index2))
        x_ticks=[-2,-1,0,1,2]
        plt.xticks(x_ticks)
        plt.yticks(x_ticks)
        plt.tick_params(direction='in') 
    



    # contour line
    n_contourline_bins=10
    limit=-ymin/n_contourline_bins    
    # colormap case 1..orange color
    if bound==2:
        plt.contour(X2, X1, M, levels = np.arange(n_contourline_bins+1)*limit+c_offset, colors='#E3BB18', linestyles='dashed',linewidth=0.7)



    ## BO points 

    c_offset=colorbar_offset[1]
    limit=1
    # Create a segmented color map with varying segment lengths
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
    }

    # Define the positions for color transitions
    positions = [0.0, 0.4, 0.7, 1.0]  # First 20% transitions quickly, remaining 60% transitions slowly
    colors = [(247/255, 216/255, 241/255), (251/255, 93/255, 88/255), (222/255, 3/255, 13/255), (89/255, 2/255, 6/255)]  #pink to# Gradient within red
            

    for pos, color in zip(positions, colors):
        r, g, b = color
        cdict['red'].append((pos, r, r))
        cdict['green'].append((pos, g, g))
        cdict['blue'].append((pos, b, b))

    custom_cmap = LinearSegmentedColormap('CustomRed', cdict); cmap=custom_cmap

    normalize = plt.Normalize(0, len(X))
    count=0
    length=len(X)
    count1=0;count2=0   

    normalize1=plt.Normalize(0,((len(X)-24)/4))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize1)
    sm.set_array([])  # You can set this to a specific array if needed
    # Add a color bar to the right of the figure

    plt.xlim(-bound-1,bound-1)
    plt.ylim(-bound-1,bound-1)

    gt_marker_size=100
    xtruez1=0
    if bound==34:
        ax.scatter(Xtrue1, Xtrue2,xtruez1, s=gt_marker_size, zorder=3, color='#12a6ce', marker='x', label='GT Max')
    #plt.xlim(0.40,0.42)
    else:
        plt.scatter(Xtrue1, Xtrue2, s=80, zorder=2.5, color='#12a6ce', marker='x', label='GT Max')
            

    folder_name='2D_plot_'+'_X_range_zoom_'+str(bound)+'Percentile_'+str(LHS_No)+'%'   
    if bound==34:
        plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.5)
        plot_filename='Figure_X'+str(index1)+'_X'+str(index2)+'wo_POINTS.svg'
        
    else:
        plt.tight_layout()
        plot_filename='Figure_X'+str(index1)+'_X'+str(index2)+'wo_POINTS.png'

    saveing_in_folder(folder_name,plot_filename)
    
   
    Y_truth=TestFunction.ackley(X,a=20,b=0.2,c=2*np.pi)
    marker_size=20
    if bound==34:
        for x,y_truth in zip(X,Y_truth):
            color=cmap(normalize(count));count1=0;                              
            if count==(len(X)-1):
                ax.scatter(x[index1-1],
                        x[index2-1],y_truth,
                        s=marker_size, zorder=2, color=[color],marker='o',label='BO Points')
            else:   
                if count<24: 
                    if count==23:            
                        ax.scatter(x[index1-1],
                                x[index2-1],y_truth,
                                s=marker_size, zorder=2, color='royalblue',label='Initial LHS')
                    else:
                        ax.scatter(x[index1-1],
                                x[index2-1],y_truth,
                                s=marker_size, zorder=2, color='royalblue')
                                                                
                else:
                    ax.scatter(x[index1-1],
                            x[index2-1],y_truth,
                            s=marker_size, zorder=2, color=[color])
            count+=1           
    else:
        for x,y_truth in zip(X,Y_truth):
            color=cmap(normalize(count));count1=0;                              
            if count==(len(X)-1):
                ax.scatter(x[index1-1],
                        x[index2-1],
                        s=marker_size, zorder=2, color=[color],marker='o',label='BO Points')
            else:   
                if count<24: 
                    if count==23:            
                        ax.scatter(x[index1-1],
                                x[index2-1],
                                s=marker_size, zorder=2, color='royalblue',label='Initial LHS')
                    else:
                        ax.scatter(x[index1-1],
                                x[index2-1],
                                s=marker_size, zorder=2, color='royalblue')
                else:
                    ax.scatter(x[index1-1],
                            x[index2-1],
                            s=marker_size, zorder=2, color=[color])
            count+=1
    
    gt_marker_size=100
    if bound==34:
        ax.scatter(Xtrue1, Xtrue2,xtruez1, s=gt_marker_size, zorder=3, color='#12a6ce', marker='x', label='GT Max')
    #plt.xlim(0.40,0.42)
    else:
        plt.scatter(Xtrue1, Xtrue2, s=80, zorder=2.5, color='#12a6ce', marker='x', label='GT Max')
            
    

    

    folder_name='2D_plot_'+'_X_range_zoom_'+str(bound)+'Percentile_'+str(LHS_No)+'%'   
    if bound==34:
        plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.5)
        plot_filename='Figure_X'+str(index1)+'_X'+str(index2)+'.svg'
    else:
        plt.tight_layout()
        plot_filename='Figure_X'+str(index1)+'_X'+str(index2)+'.png'
    
    #plot_filename='Figure_X'+str(index1)+'_X'+str(index2)+'.svg'
    saveing_in_folder(folder_name,plot_filename)
    plt.close()


    

def predicted_model_perLHS(model_X,model,BO_model,Y_mean,Y_std,ymin,noise,LHS_No):
    X,x1,x2,x3,x4,x5,x6,n=meshgrid()

    X=torch.tensor(X, dtype=float)

    predicted=(model(X).mean)
    Y_pred=Norm_Dnorm.destandardizer(predicted,Y_mean,Y_std)
    Y_pred=Y_pred.detach().numpy();

    Y_pred=np.array(Y_pred)
    X,x1,x2,x3,x4,x5,x6,n=meshgrid()
    X=np.array(X)

    idx = np.argmax(Y_pred)
    #X=Norm_Dnorm.denormalizer(X, xmax, xmin);
    X1max, X2max, X3max, X4max, X5max, X6max = X[idx, 0], X[idx, 1], X[idx, 2], X[idx, 3], X[idx, 4], X[idx, 5]
    X1true, X2true, X3true, X4true, X5true, X6true = 0,0,0,0,0,0
    X1true1, X2true1, X3true1, X4true1, X5true1, X6true1 = 0,0,0,0,0,0

    

    Ycube = np.reshape(Y_pred, (n, n, n, n, n, n))
    M12 = np.max(Ycube, axis=(2, 3, 4, 5)); M12=squeeze(M12)

    # M13
    M13 = np.max(Ycube, axis=(1, 3, 4, 5));M13 = np.squeeze(M13)
    M14 = np.max(Ycube, axis=(1, 2, 4, 5));M14 = np.squeeze(M14)
    M15 = np.max(Ycube, axis=(1, 2, 3, 5));M15 = np.squeeze(M15)
    M16 = np.max(Ycube, axis=(1, 2, 3, 4));M16 = np.squeeze(M16)
    M23 = np.max(Ycube, axis=(0, 3, 4, 5));M23 = np.squeeze(M23)
    M24 = np.max(Ycube, axis=(0, 2, 4, 5));M24 = np.squeeze(M24)
    M25 = np.max(Ycube, axis=(0, 2, 3, 5));M25 = np.squeeze(M25)

    M26 = np.max(Ycube, axis=(0, 2, 3, 4));M26 = np.squeeze(M26)
    M34 = np.max(Ycube, axis=(0, 1, 4, 5));M34 = np.squeeze(M34)
    M35 = np.max(Ycube, axis=(0, 1, 3, 5));M35 = np.squeeze(M35)
    M36 = np.max(Ycube, axis=(0, 1, 3, 4));M36 = np.squeeze(M36)

    M45 = np.max(Ycube, axis=(0, 1, 2, 5));M45 = np.squeeze(M45)
    M46 = np.max(Ycube, axis=(0, 1, 2, 4));M46 = np.squeeze(M46)
    M56 = np.max(Ycube, axis=(0, 1, 2, 3));M56 = np.squeeze(M56)

  
                 

        # 2D Heat maps


    bounds=[34,2]
    for bound in bounds:
        
       
    # 3D Surface plot
   
        
        subplot_row=1;subplot_column=1 
        create_subplot_surface(1, subplot_row,subplot_column,x1, x2, M12, model_X,X1max, X2max, X1true, X2true,X1true1, X2true1,ymin,LHS_No,noise,bound,index1=1,index2=2)
        create_subplot_surface(2, subplot_row,subplot_column,x1, x3, M13, model_X,X1max, X3max, X1true, X3true,X1true1, X3true1,ymin,LHS_No,noise,bound,index1=1,index2=3)
        create_subplot_surface(3, subplot_row,subplot_column,x1, x4, M14, model_X,X1max, X4max, X1true, X4true,X1true1, X4true1,ymin,LHS_No,noise,bound,index1=1,index2=4)
        create_subplot_surface(4, subplot_row,subplot_column,x1, x5, M15, model_X,X1max, X5max, X1true, X5true,X1true1, X5true1,ymin,LHS_No,noise,bound,index1=1,index2=5)
        create_subplot_surface(5, subplot_row,subplot_column,x1, x6, M16, model_X,X1max, X6max, X1true, X6true,X1true1, X6true1,ymin,LHS_No,noise,bound,index1=1,index2=6)


        # if bound==34:
        #     #movie plot starting
        #     xmax=32.768;xmin=-32.768;ymax=0
        #     movie_plotting.movie_plot(1, x1, x2, BO_model, model_X,xmax, xmin, X1true, X2true,ymax,ymin,bound,Y_mean,Y_std)
