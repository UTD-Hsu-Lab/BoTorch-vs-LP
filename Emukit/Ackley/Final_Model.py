import numpy as np
import time
import math
from scipy.stats import qmc
from emukit.core import ParameterSpace, ContinuousParameter
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import GPy
import csv
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop

from emukit.core.initial_designs.random_design import RandomDesign
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import subprocess
import TestFunction
import Norm_Dnorm
import Noise_
import record_data_log_file
import movie_plotting
import os



def plot_contour_per_LHS(nvar,f_obj,ymax,ymin,xmax,xmin,X,Y,run,std_noise,jitter,percentile_no,model):
        
    parity_plot(nvar,f_obj,ymax,ymin,X,Y,run,std_noise,jitter,xmax,xmin,percentile_no) 
    
    y_pred=predicted_objective_function(nvar,f_obj,ymax,ymin,xmax,xmin,X,Y,run,std_noise,jitter,percentile_no,model)

def saveing_in_folder(folder_name,plot_filename):
    folder_path = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    plot_filepath = os.path.join(folder_path, plot_filename)
    plt.savefig(plot_filepath,transparent='clear',facecolor='white',bbox_inches='tight')

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

def meshgrid(n):
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

    X1, X2, X3, X4, X5, X6 = np.meshgrid(x1, x2, x3, x4, x5, x6, indexing='ij')
    return X1, X2, X3, X4, X5, X6,x1,x2,x3,x4,x5,x6

def predicted_objective_function(nvar,f_obj,ymax,ymin,xmax,xmin,model_X,Y,run,std_noise,jitter,percentile_no,model):


    n= 11
    X1,X2,X3,X4,X5,X6,x1,x2,x3,x4,x5,x6=meshgrid(n)
    X_ = np.column_stack([X1.ravel(), X2.ravel(), X3.ravel(), X4.ravel(), X5.ravel(), X6.ravel()])


    # Compute y_pred using the gp model
    y_pred, ___ = f_obj(X_)    
    y_pred=Norm_Dnorm.denormalizer(-y_pred, ymax, ymin);y_pred = np.array(y_pred)       
    X = np.array(X_)

    idx = np.argmax(y_pred)
    X_=Norm_Dnorm.denormalizer(X_, xmax, xmin);
    X1max, X2max, X3max, X4max, X5max, X6max = X_[idx, 0], X_[idx, 1], X_[idx, 2], X_[idx, 3], X_[idx, 4], X_[idx, 5]
    X1true, X2true, X3true, X4true, X5true, X6true = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    
    #predicted model for corresponding X  variable

    Ycube = np.reshape(y_pred, (n, n, n, n, n, n))
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

    X_zoom_range=[33,4]
    for bound in X_zoom_range:
        # Create subplots
        index2=0;index1=0 
        
        create_subplot(1, x1, x2, M12, model_X,Y,X1max, X2max, X1true, X2true,ymax,ymin,bound,percentile_no,index1=1,index2=2)
        create_subplot(2, x1, x3, M13, model_X,Y,X1max, X3max, X1true, X3true,ymax,ymin,bound,percentile_no,index1=1,index2=3)
        create_subplot(3, x1, x4, M14, model_X,Y,X1max, X4max, X1true, X4true,ymax,ymin,bound,percentile_no,index1=1,index2=4)
        create_subplot(4, x1, x5, M15, model_X,Y,X1max, X5max, X1true, X5true,ymax,ymin,bound,percentile_no,index1=1,index2=5)
        create_subplot(5, x1, x6, M16, model_X,Y,X1max, X6max, X1true, X6true,ymax,ymin,bound,percentile_no,index1=1,index2=6)
 

        #movie plot starting
        if bound==33 or bound==4:
            movie_plotting.movie_plot(1, x1, x2, model, model_X,Y,xmax, xmin, X1true, X2true,ymax,ymin,bound,percentile_no,std_noise,index1=1,index2=2)

 # Helper function to create each subplot
def create_subplot(subplot_num, x1, x2, M, X,Y,Xmax1,Xmax2, Xtrue1,Xtrue2,ymax,ymin,bound,percentile_no,index1,index2): 
    figwidth=2.2;figheight=2
    plt.style.use('seaborn-paper')
    fig1 = plt.figure(figsize=(figwidth,figheight))
    if bound==33:
        ax=fig1.add_subplot(1, 1, 1,projection='3d',computed_zorder=False)
    else:
        ax=fig1.add_subplot(1, 1, 1)


    X1, X2 = np.meshgrid(x1, x2)
    xmax=32.768;xmin=-32.768;
    X1=Norm_Dnorm.denormalizer(X1, xmax, xmin);X2=Norm_Dnorm.denormalizer(X2, xmax, xmin);
    colorbar_offset = [ymin, 0, 0]

    n_bins=2585
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
    start_color = (0.0588, 0.5529, 0.0314);end_color = (0.937,0.984,0.169) 
    fast_color = (start_color[0] * 0.2, start_color[1] * 0.2, start_color[2] * 0.2)  # 20% of start color
    colors = [fast_color, start_color, end_color]

    for pos, color in zip(positions, colors):
        r, g, b = color
        cdict['red'].append((pos, r, r))
        cdict['green'].append((pos, g, g))
        cdict['blue'].append((pos, b, b))

    custom_cmap = LinearSegmentedColormap('CustomColormap', cdict)
    cmap = custom_cmap 
    
    if bound==33:
        cbar= ax.plot_surface(X2, X1, M,cmap=cmap,zorder=1)
    else:
        plt.contourf(X2, X1, M, 50, levels = np.arange(n_bins+1)*limit+c_offset, cmap=cmap)
    
    
    zticks=[-20,-10,0]
    if bound==4:
        plt.xlabel('x'+str(index1))
        plt.ylabel('x'+str(index2))
        x_ticks=[-4,-2,0,2,4]
        plt.xticks(x_ticks)
        plt.yticks(x_ticks)
        plt.tick_params(direction='in') 
        
    elif bound==33:
        plt.xlabel('x'+str(index1),labelpad=0.1)
        plt.ylabel('x'+str(index2),labelpad=0.1)
        x_ticks=[-30,0,30]
        plt.xticks(x_ticks)
        plt.yticks(x_ticks)
        ax.set_zticks(zticks)            
        plt.tick_params(direction='in',pad=0.1) 
    
    # contour line for 2D heat maps
    n_contourline_bins=10
    limit=-ymin/n_contourline_bins
    if bound==4:
        plt.contour(X2, X1, M, levels = np.arange(n_contourline_bins+1)*limit+c_offset, colors='#E3BB18', linestyles='dashed',linewidth=0.7)
     
    #  BO Points pROJECTIONS    
    # Create a segmented color map with varying segment lengths
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
    }
    # Define the positions for color transitions
    positions = [0.0, 0.4, 0.7, 1.0]  
    # Define colors and their corresponding positions for transition    
    colors = [(247/255, 216/255, 241/255), (251/255, 93/255, 88/255), (222/255, 3/255, 13/255), (89/255, 2/255, 6/255)]  #pink to# Gradient within red
    for pos, color in zip(positions, colors):
        r, g, b = color
        cdict['red'].append((pos, r, r))
        cdict['green'].append((pos, g, g))
        cdict['blue'].append((pos, b, b))

    custom_cmap = LinearSegmentedColormap('CustomRed', cdict);cmap=custom_cmap        

    #pOINTS PROJECTIONS TO HEAT MAPS
    normalize = plt.Normalize(0, len(X))
    count=0;length=len(X);count1=0;count2=0
    Y_truth=TestFunction.ackley(X,a=20,b=0.2,c=2*np.pi)
    marker_size=20

    ## POINTS PROJECTION TO 3D HEAT MAP
    if bound==33:
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
    ## POINTS PROJECTION TO 2D HEAT MAP         
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

    gt_marker_size=80;
    xtruez1=0
    if bound==33:
        ax.scatter(Xtrue1, Xtrue2,xtruez1, s=gt_marker_size, zorder=3, color='#12a6ce', marker='x', label='GT Max')
    else:
        plt.scatter(Xtrue1, Xtrue2, s=80, zorder=2.5, color='#12a6ce', marker='x', label='GT Max')
    

    
    plt.xlim(-bound,bound)
    plt.ylim(-bound,bound)
    
    folder_name='2D_plot_'+'_X_range_zoom_'+str(bound)+'Percentile_'+str(percentile_no)+'%'   
    if bound==33:
        plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.5)
        plot_filename='Figure_X'+str(index1)+'_X'+str(index2)+'.svg'
    else:
        plt.tight_layout()
        plot_filename='Figure_X'+str(index1)+'_X'+str(index2)+'.png'
    
    saveing_in_folder(folder_name,plot_filename)
    plt.close()


def parity_plot(nvar,f_obj,ymax,ymin,X,Y,run,std_noise,jitter,xmax,xmin,percentile_no):


    ymax=0;ymin=-22.3    
    X_=X;a=20;b=0.2;c=2*np.pi
    Y_truth=(TestFunction.ackley(X_,a,b,c));Y_truth=np.array(Y_truth).reshape(len(Y_truth),1)   
    Y_pred=np.array(Y)

    filename='parity_data.csv'
    record_data_log_file.record_parity_plot_data(X,Y_truth,Y_pred,filename,percentile_no)
    
    # Reset the limits
    fig1 = plt.figure(1)
    fig1.set_size_inches(2.2, 2)
    plt.style.use('seaborn-paper')
    ax = plt.gca()
    bounds=-22,0
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    yticks=[-20.0,-15.0,-10.0,-5.0,0.0]
    ax.set_yticks(yticks)
    ax.set_xticks(yticks)

    plt.scatter(Y_truth,Y_pred,s = 30, facecolors='#5171F4', alpha = 0.5, edgecolor = 'blue')
    x = np.linspace(min(np.min(Y_truth), np.min(Y_pred)), max(np.max(Y_truth), np.max(Y_pred)), 100)
    plt.plot(x, x, 'r--')  # Red dashed line indicating y = x


   
    # Calculate Statistics of the Parity Plot 
    
    x=Y_truth.flatten();y=Y_pred.flatten()
    mean_abs_err = np.mean(np.abs(x-y))
    rmse = np.sqrt(np.mean((x-y)**2))
    rmse_std = rmse / np.std(y)
    
    #Title and labels 
    plt.xlabel('Ground Truth')
    plt.ylabel('Model Prediction')    
    plt.tick_params(direction='in') 
    
    plt.tight_layout()    
    plot_filename='parity_plot'+'Percentile_'+str(percentile_no)+'.svg'   
    folder_name='parity_plot'
    saveing_in_folder(folder_name,plot_filename)
    plt.close()
    