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
from emukit.bayesian_optimization.acquisitions import NegativeLowerConfidenceBound, ExpectedImprovement
from emukit.core.initial_designs.random_design import RandomDesign
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import subprocess
import TestFunction
import Norm_Dnorm
import Noise_
import record_data_log_file
import movie_plotting
import os


np.seterr(divide = 'ignore')

def plot_contour_per_LHS(nvar,f_obj,ymax,ymin,xmax,xmin,X,Y,std_noise,beta,percentile_no,model):
        
    parity_plot(nvar,f_obj,ymax,ymin,X,Y,std_noise,beta,xmax,xmin,percentile_no)   

    predicted_objective_function(nvar,f_obj,ymax,ymin,xmax,xmin,X,Y,std_noise,beta,percentile_no,model)

def saveing_in_folder(folder_name,plot_filename):
    folder_path = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    plot_filepath = os.path.join(folder_path, plot_filename)
    plt.savefig(plot_filepath,transparent='clear',facecolor='white',bbox_inches='tight')

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

def predicted_objective_function(nvar,f_obj,ymax,ymin,xmax,xmin,model_X,Y,std_noise,beta,percentile_no,model):


    n= 11
    X1,X2,X3,X4,X5,X6,x1,x2,x3,x4,x5,x6=meshgrid(n)
    
    # stacking into one variable for simplification  
    X_ = np.column_stack([X1.ravel(), X2.ravel(), X3.ravel(), X4.ravel(), X5.ravel(), X6.ravel()])


    # Compute y_pred using the gp model
    y_pred, ___ = f_obj(X_)
    y_pred=-Norm_Dnorm.denormalizer(y_pred, ymax, ymin);y_pred = np.array(y_pred)       
    X = np.array(X_)

    idx = np.argmax(y_pred)
    X1max, X2max, X3max, X4max, X5max, X6max = X_[idx, 0], X_[idx, 1], X_[idx, 2], X_[idx, 3], X_[idx, 4], X_[idx, 5]
    X1true, X2true, X3true, X4true, X5true, X6true = 0.20169, 0.150011, 0.476874, 0.275322, 0.311652, 0.65730 #hartmann 1st max coordinate
    X1true1, X2true1, X3true1, X4true1, X5true1, X6true1 = 0.40460235,0.88231846,0.81636135,0.57388939,0.14649013,0.03864136 #hartmann 2nd max coordinate



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



    

    ##individual_plot
    
    
    plot_condition=0;
    subplot_row=1;subplot_column=1   
    
    create_subplot(1, subplot_row,subplot_column,x1, x2, M12, model_X,Y,X1max, X2max, X1true, X2true,X1true1, X2true1,ymax,plot_condition,percentile_no,index1=1,index2=2)
    create_subplot(1, subplot_row,subplot_column,x1, x3, M13, model_X,Y,X1max, X3max, X1true, X3true,X1true1, X3true1,ymax,plot_condition,percentile_no,index1=1,index2=3)
    create_subplot(1, subplot_row,subplot_column,x1, x4, M14, model_X,Y,X1max, X4max, X1true, X4true,X1true1, X4true1,ymax,plot_condition,percentile_no,index1=1,index2=4)
    create_subplot(1, subplot_row,subplot_column,x1, x5, M15, model_X,Y,X1max, X5max, X1true, X5true,X1true1, X5true1,ymax,plot_condition,percentile_no,index1=1,index2=5)
    create_subplot(1, subplot_row,subplot_column,x1, x6, M16, model_X,Y,X1max, X6max, X1true, X6true,X1true1, X6true1,ymax,plot_condition,percentile_no,index1=1,index2=6)
    create_subplot(1, subplot_row,subplot_column,x1, x6, M16, model_X,Y,X1max, X6max, X1true, X6true,X1true1, X6true1,ymax,plot_condition,percentile_no,index1=6,index2=6)
    
    #combined plot
    figwidth=2.2;figheight=2;
    plot_condition=1;
    fig1 = plt.figure(1)
    fig1.set_size_inches(figwidth,figheight)
    plt.style.use('seaborn-paper')
    subplot_row=3;subplot_column=5
    create_subplot(1, subplot_row,subplot_column,x1, x2, M12, model_X,Y,X1max, X2max, X1true, X2true,X1true1, X2true1,ymax,plot_condition,percentile_no,index1=1,index2=2)
    create_subplot(2, subplot_row,subplot_column,x1, x3, M13, model_X,Y,X1max, X3max, X1true, X3true,X1true1, X3true1,ymax,plot_condition,percentile_no,index1=1,index2=3)
    create_subplot(3, subplot_row,subplot_column,x1, x4, M14, model_X,Y,X1max, X4max, X1true, X4true,X1true1, X4true1,ymax,plot_condition,percentile_no,index1=1,index2=4)
    create_subplot(4, subplot_row,subplot_column,x1, x5, M15, model_X,Y,X1max, X5max, X1true, X5true,X1true1, X5true1,ymax,plot_condition,percentile_no,index1=1,index2=5)
    create_subplot(5, subplot_row,subplot_column,x1, x6, M16, model_X,Y,X1max, X6max, X1true, X6true,X1true1, X6true1,ymax,plot_condition,percentile_no,index1=1,index2=6)

    
    
    create_subplot(6, subplot_row,subplot_column,x2, x3, M23, model_X,Y,X2max, X3max, X2true, X3true,X2true1, X3true1,ymax,plot_condition,percentile_no,index1=2,index2=3)
    create_subplot(7, subplot_row,subplot_column,x2, x4, M24, model_X,Y,X2max, X4max, X2true, X4true,X2true1, X4true1,ymax,plot_condition,percentile_no,index1=2,index2=4)
    create_subplot(8, subplot_row,subplot_column,x2, x5, M25, model_X,Y,X2max, X5max, X2true, X5true,X2true1, X5true1,ymax,plot_condition,percentile_no,index1=2,index2=5)
    create_subplot(9, subplot_row,subplot_column,x2, x6, M26, model_X,Y,X2max, X6max, X2true, X6true,X2true1, X6true1,ymax,plot_condition,percentile_no,index1=2,index2=6)

    create_subplot(10, subplot_row,subplot_column,x3, x4, M34, model_X,Y,X3max, X4max, X3true, X4true,X3true1, X4true1,ymax,plot_condition,percentile_no,index1=3,index2=4)
    create_subplot(11, subplot_row,subplot_column,x3, x5, M35, model_X,Y,X3max, X5max, X3true, X5true,X3true1, X5true1,ymax,plot_condition,percentile_no,index1=3,index2=5)
    create_subplot(12, subplot_row,subplot_column,x3, x6, M36, model_X,Y,X3max, X6max, X3true, X6true,X3true1, X6true1,ymax,plot_condition,percentile_no,index1=3,index2=6)

    create_subplot(13, subplot_row,subplot_column,x4, x5, M45, model_X,Y,X4max, X5max, X4true, X5true,X4true1, X5true1,ymax,plot_condition,percentile_no,index1=4,index2=5)
    create_subplot(14, subplot_row,subplot_column,x4, x6, M46, model_X,Y,X4max, X6max, X4true, X6true,X4true1, X6true1,ymax,plot_condition,percentile_no,index1=4,index2=6)

    create_subplot(15, subplot_row,subplot_column,x5, x6, M56, model_X,Y,X5max, X6max, X5true, X6true,X5true1, X6true1,ymax,plot_condition,percentile_no,index1=5,index2=6)

    plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.5)
    folder_name='2D_plot_'+'_Percentile_'+str(percentile_no)+'%'   
    plot_filename='X15.svg'
    saveing_in_folder(folder_name,plot_filename)
    plt.close()
    
    
    
    #movie plot starting
    bound=1
    movie_plotting.movie_plot(beta, x1, x2, model, model_X,Y,xmax, xmin, X1true, X2true,ymax,ymin,bound,percentile_no,std_noise,index1=1,index2=2)
    movie_plotting.movie_plot(beta, x1, x3, model, model_X,Y,xmax, xmin, X1true, X2true,ymax,ymin,bound,percentile_no,std_noise,index1=1,index2=3)
    movie_plotting.movie_plot(beta, x1, x4, model, model_X,Y,xmax, xmin, X1true, X2true,ymax,ymin,bound,percentile_no,std_noise,index1=1,index2=4)
    movie_plotting.movie_plot(beta, x1, x5, model, model_X,Y,xmax, xmin, X1true, X2true,ymax,ymin,bound,percentile_no,std_noise,index1=1,index2=5)
    movie_plotting.movie_plot(beta, x1, x6, model, model_X,Y,xmax, xmin, X1true, X2true,ymax,ymin,bound,percentile_no,std_noise,index1=1,index2=6)
    

# Helper function to create each subplot
def create_subplot(subplot_num, subplot_row,subplot_column, x1, x2, M, X,Y,Xmax1,Xmax2, Xtrue1,Xtrue2,X1true1,X1true2,ymax,plot_condition,percentile_no,index1,index2):
    if plot_condition==0:
        figwidth=2.2;figheight=2;
        fig1=plt.figure(figsize=(figwidth,figheight))
        plt.style.use('seaborn-paper')
        ax=fig1.add_subplot(subplot_row, subplot_column, subplot_num,projection='3d', computed_zorder=False)
    else:
        ax=plt.subplot(subplot_row, subplot_column, subplot_num,projection='3d', computed_zorder=False)

    #defining corresponding input X varaibles
    X1, X2 = np.meshgrid(x1, x2)
    colorbar_offset = [0, 0, 0]
    n_bins = 2585  # Number of bins in the colormap
    limit=ymax/n_bins
    c_offset=colorbar_offset[0]


      
    if index1!=6:
        #plt.contourf(X2, X1, M, 50, levels = np.arange(n_bins+1)*limit+c_offset, cmap=cmap)
        cbar=ax.plot_surface(X2, X1, M,cmap='plasma',zorder=1)
    else:
        plt.contourf(X2, X1, M, 50, levels = np.arange(n_bins+1)*limit+c_offset, cmap='plasma')    
    

    xyticks=[0.0,0.5,1.0]
    zticks=[0,1,2,3]    
        
    plt.xlabel('x'+str(index1), labelpad=0.1);plt.xticks(xyticks)
    plt.ylabel('x'+str(index2), labelpad=0.1);plt.yticks(xyticks) 
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zticks(zticks) 
    plt.tick_params(direction='in',pad=0.1)
    
    ### colorbar plotting      
    # cb = plt.colorbar(pad=0.8)
    # ticks=[0.0,0.4,0.8,1.2,1.6,2.0,2.4,2.8,3.2]
    # cb.set_ticks(ticks)                  
    # cb.set_label('Max(μ_D($\mathbf{X}$))', labelpad=0)            
        

              
                
   ## BO points color map............................
        
    c_offset=colorbar_offset[1]
    limit=1
    # Create a segmented color map with varying segment lengths
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
    }

    # Define the positions for color transitions
    positions = [0.0, 0.33, 0.67, 1.0]

    D=[70, 254, 75  ];C=[70, 254, 196    ];B=[70, 191, 254   ];A=[ 70, 85, 254 ]
    # Define colors and their corresponding positions for transition
    colors = [(A[0]/255,A[1]/255,A[2]/255), (B[0]/255,B[1]/255,B[2]/255), (C[0]/255,C[1]/255,C[2]/255), (D[0]/255,D[1]/255,D[2]/255)]
                
    for pos, color in zip(positions, colors):
        r, g, b = color
        cdict['red'].append((pos, r, r))
        cdict['green'].append((pos, g, g))
        cdict['blue'].append((pos, b, b))

    custom_cmap = LinearSegmentedColormap('CustomRed', cdict);cmap=custom_cmap  
    normalize = plt.Normalize(0, len(X))
    count=0
    length=len(X)
    count1=0;count2=0   

    normalize1=plt.Normalize(0,50)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize1)
    sm.set_array([])  # You can set this to a specific array if needed


    ##  Add a iteration color bar to the right of the figure
    
    # cbar = plt.colorbar(sm, pad=0.05)
    # cbar.set_label('Iteration', labelpad=0)            
    # num_ticks = 2  # Adjust the number of ticks as needed
    # cbar.locator=MaxNLocator(integer=True, nbins=num_ticks)
    # #cbar.ax.tick_params(labelsize=15)
    # cbar.update_ticks()


    ## BO points projection................................
    marker_size=20
    height_for_visibility=0.0
    Y_truth=TestFunction.hartmann(X)
    for x,y_truth in zip(X,Y_truth):            
        color=cmap(normalize(count-24));count1=0;  
        if count==(len(X)-1):
            ax.scatter(x[index1-1], x[index2-1],y_truth,
                    s=marker_size, zorder=2, color=[color],marker='o',label='BO Points')
        else:  
            if count<24:
                
                ax.scatter(x[index1-1],
                    x[index2-1],y_truth,
                    s=marker_size, zorder=2, color='red')  #medium green
                if count==23:
                    ax.scatter(x[index1-1],x[index2-1],y_truth,s=marker_size, zorder=2, color='red',label='LHS Points')  #medium green

            else:  
                ax.scatter(x[index1-1],
                        x[index2-1],y_truth,
                        s=marker_size, zorder=2, color=[color],alpha=1.0)
        count+=1

    
    
    gt_marker_size=80;        
    xtruez1=ymax; xtruez2=3.22231 #(2nd true max)
    ax.scatter(Xtrue1, Xtrue2,xtruez1, s=gt_marker_size-30, zorder=3, color='black', marker='x', label='GT Max')
    ax.scatter(X1true1, X1true2,xtruez2, s=gt_marker_size, zorder=3, color='#C11B17', marker='*', label='2nd True Max')


    ## legend adding
               
        # legend = plt.legend()
        # legend.get_frame().set_alpha(0.6)
        # legend_font = { 'size': 18}
        # #plt.legend(prop=legend_font,loc='upper left', bbox_to_anchor=(1.2, 1.25))
        # plt.legend(loc='upper left', bbox_to_anchor=(1.8, 1.0))

    ## individual plot saving
    if plot_condition==0:

        plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.5)
        folder_name='2D_plot_'+'_Percentile_'+str(percentile_no)+'%'   
        plot_filename='X'+str(index1)+'_X'+str(index2)+'.svg'
        saveing_in_folder(folder_name,plot_filename)
        plt.close()
 

def parity_plot(nvar,f_obj,ymax,ymin,X,Y,std_noise,beta,rmax,rmin,percentile_no):


    
    X_=X;
    Y_truth=TestFunction.hartmann(X_);Y_truth=np.array(Y_truth).reshape(len(Y_truth),1)   
    Y_pred=np.array(-Y).reshape(len(Y_truth),1) 

    print(Y_truth,Y_pred)

    filename='parity_data.csv'
    record_data_log_file.record_parity_plot_data(X,Y_truth,Y_pred,filename,percentile_no)
    
    # Reset the limits
    fig1 = plt.figure(1)
    fig_width=2.2;fig_height=2
    fig1.set_size_inches(fig_width, fig_height)
    plt.style.use('seaborn-paper')
    ax = plt.gca()
    bounds=(0,3.5)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    
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
    ticks=[0,1,2,3,4]
    plt.xticks(ticks)
    plt.yticks(ticks)  
    plt.tick_params(direction='in') 
    
    #legend_font = { 'size': 15}
    #plt.legend(prop=legend_font,loc='best')
    plt.tight_layout()

      
    folder_name='parity_plot' 
    plot_filename='parity_plot'+'Percentile_'+str(percentile_no)+'%.svg'
    saveing_in_folder(folder_name,plot_filename)
    plt.close()
    
    
