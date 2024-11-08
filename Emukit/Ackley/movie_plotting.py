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
from mpl_toolkits.mplot3d import Axes3D
import record_data_log_file
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import Final_Model


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

def model_prediction_per_iteration(model,ymax,ymin,xmax,xmin,frame):
    n=11
    X1, X2, X3, X4, X5, X6,x1,x2,x3,x4,x5,x6=Final_Model.meshgrid(n)
    X_ = np.column_stack([X1.ravel(), X2.ravel(), X3.ravel(), X4.ravel(), X5.ravel(), X6.ravel()])


    # Compute y_pred using the hart6D function
    f_obj=model[frame]
    y_pred, ___ = f_obj(X_)    
    y_pred=Norm_Dnorm.denormalizer(-y_pred, ymax, ymin);y_pred = np.array(y_pred)       
    X = np.array(X_)

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

def update(frame,X1,X2,f_obj,X,xmax,xmin,ymax,ymin,bound,frame_length,ax):
        
    #prediction per iteration
    M12,M13,M14,M15,M16,X1max,X2max,X3max,X4max,X5max,X6max,X1true,X2true,X3true,X4true,X5true,X6true=model_prediction_per_iteration(f_obj,ymax,ymin,xmax,xmin,frame)
    
    
    
    index1=1;index2=2
    colorbar_offset = [ymin, 0, 0]
    n_bins = 2585  # Number of bins in the colormap
    limit=-ymin/2585
    c_offset=colorbar_offset[0]
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
    }
    # Define the positions for color transitions
    positions = [0.0, 0.20, 1.0]  # First 20% transitions quickly, remaining 80% transitions slowly
    # Define colors and their corresponding positions for transition
    start_color = (0.0588, 0.5529, 0.0314);end_color = (0.937,0.984,0.169) # yellow
    fast_color = (start_color[0] * 0.2, start_color[1] * 0.2, start_color[2] * 0.2)  # 20% of start color
    colors = [fast_color, start_color, end_color]
    for pos, color in zip(positions, colors):
        r, g, b = color
        cdict['red'].append((pos, r, r))
        cdict['green'].append((pos, g, g))
        cdict['blue'].append((pos, b, b))
    custom_cmap = LinearSegmentedColormap('CustomColormap', cdict);cmap = custom_cmap 

    if bound==4:
        plt.contourf(X2, X1, M12, 50, levels = np.arange(n_bins+1)*limit+c_offset, cmap=cmap)
    else:        
        cbar=ax.plot_surface(X2, X1, M12,cmap=cmap,zorder=1)
    plt.xlabel('x'+str(index1), labelpad=0.1)
    plt.ylabel('x'+str(index2), labelpad=0.1)
    zticks=[-20,-10,0]
    if bound==33:    
        x_ticks=[-30,0,30]
        plt.xticks(x_ticks)
        plt.yticks(x_ticks)
        ax.set_zticks(zticks)     
    else:
        x_ticks=[-4,-2,0,2,4]
        plt.xticks(x_ticks)
        plt.yticks(x_ticks)    
    plt.tick_params(direction='in',pad=0.1) 
    
    #                                    contour line
    n_contourline_bins=10
    limit=-ymin/n_contourline_bins
    if bound==4:
        plt.contour(X2, X1, M12, levels = np.arange(n_contourline_bins+1)*limit+c_offset, colors='#E3BB18', linestyles='dashed',linewidth=0.7)

    #  BO POINTS PROJECTION
    # Create a segmented color map with varying segment lengths
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
    }
    positions = [0.0, 0.4, 0.7, 1.0]  # First 40% transitions quickly, remaining 60% transitions slowly
    colors = [(247/255, 216/255, 241/255), (251/255, 93/255, 88/255), (222/255, 3/255, 13/255), (89/255, 2/255, 6/255)]  #pink to# Gradient within red
    for pos, color in zip(positions, colors):
        r, g, b = color
        cdict['red'].append((pos, r, r))
        cdict['green'].append((pos, g, g))
        cdict['blue'].append((pos, b, b))
    custom_cmap = LinearSegmentedColormap('CustomRed', cdict);cmap=custom_cmap  
    normalize=plt.Normalize(0,frame_length)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    sm.set_array([])  # You can set this to a specific array if needed
    
    plt.xlim(-bound,bound)
    plt.ylim(-bound,bound)

    ##  LHS  POINTS PROJECTION
    n_lhs=24;
    marker_size=20
    Y_truth=TestFunction.ackley(X,a=20,b=0.2,c=2*np.pi)
    for i in range(n_lhs):
        x=X[i]
        y_truth=Y_truth[i]
        if bound==33:
            ax.scatter(x[index1-1],x[index2-1],y_truth,s=marker_size, zorder=2, color='royalblue',marker='o')
        else:
            plt.scatter(x[index1-1],x[index2-1],s=marker_size, zorder=2, color='royalblue',marker='o')
    if bound==33:
        ax.scatter(x[index1-1],x[index2-1],y_truth,s=marker_size, zorder=2, color='royalblue',marker='o',label='Initial LHS')
    else:
        plt.scatter(x[index1-1],x[index2-1],s=marker_size, zorder=2, color='royalblue',marker='o',label='Initial LHS')

    plt.title(f'Iteration {frame+1}')
    index_start=24;index_end=int(24+(frame)*4)+4

    #  BO reaminging  POINTS PROJECTION
    color=cmap(normalize(frame));ctr=0    
    for i in range(index_start,index_end,1):
        temp=(i)%4
        if temp==0:
            index=int(float(i/4))
            color=cmap(normalize(index))
        x=X[i]  
        y=Y_truth[i]        
        if bound==33:            
            ax.scatter(x[index1-1],x[index2-1],y,s=marker_size, zorder=2, color=[color])
        else:
            plt.scatter(x[index1-1],x[index2-1],s=marker_size, zorder=2, color=[color])
    if bound==33:
        ax.scatter(x[index1-1],x[index2-1],y,s=marker_size, zorder=2, color=[color],marker='o',label='BO Points')
    else:
        plt.scatter(x[index1-1],x[index2-1],s=marker_size, zorder=2, color=[color],marker='o',label='BO Points')
    

    gt_marker_size=80;xtruez1=0
    if bound==33:
        ax.scatter(X1true, X1true,xtruez1, s=gt_marker_size, zorder=3, color='#12a6ce', marker='x', label='GT Max')
    else:
        plt.scatter(X1true, X2true, s=80, zorder=5, color='#12a6ce', marker='x', label='True Max')                
    print('Frame No==',frame)
    
    # legend = plt.legend()
    # legend.get_frame().set_alpha(0.6)
    # if index1==1 and index2==2:
    #     plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.35))        
    # else:
    #     plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.35))
    plt.tight_layout()


def movie_plot(beta, x1, x2, f_obj, model_X,X_new_dn,xmax, xmin, X1true, X2true,ymax,ymin,bound,percentile_no,std_noise,index1,index2):
    fig, ax = plt.subplots()
    X1, X2 = np.meshgrid(x1, x2)
    xmax=32.768;xmin=-32.768
    X1=Norm_Dnorm.denormalizer(X1, xmax, xmin);X2=Norm_Dnorm.denormalizer(X2, xmax, xmin);
    
    figwidth=2.2
    figheight=2
    frame_length=int((len(model_X)-24)/4) 
    folder_name='Movie_plot_'+'_Percentile_'+str(percentile_no)+'%'   
    for frame_number in range(frame_length):
        temp=int((frame_number+1)%4)
        if frame_number<=20 or frame_number==(frame_length-1) or frame_number==22 or frame_number==28 or frame_number==32:
            fig1 = plt.figure(figsize=(figwidth,figheight))
            if bound==33:
                ax=fig1.add_subplot(1, 1, 1,projection='3d',computed_zorder=False)
            plt.style.use('seaborn-paper')
            update(frame_number, X1, X2, f_obj, model_X, xmax, xmin, ymax, ymin, bound, frame_length,ax)
            plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.5)                        
            plot_filename='frame_'+str(frame_number)+'.png'
            saveing_in_folder(folder_name,plot_filename)
            plt.close()
        elif frame_number>20 and temp==0:
            fig1 = plt.figure(figsize=(figwidth,figheight))
            if bound==33:
                ax=fig1.add_subplot(1, 1, 1,projection='3d',computed_zorder=False)
            plt.style.use('seaborn-paper')
            update(frame_number, X1, X2, f_obj, model_X, xmax, xmin, ymax, ymin, bound, frame_length,ax)
            plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.5)
            plot_filename='frame_'+str(frame_number)+'.png'
            saveing_in_folder(folder_name,plot_filename)
            plt.close()




    directory_path = os.path.join(os.getcwd(), folder_name)
    file_list = os.listdir(directory_path)
    file_list = [file for file in file_list if re.match(r'frame_\d+\.png', file)]
    file_list = sorted(file_list, key=lambda x: int(re.search(r'(\d+)', x).group(1)))
    def animate(frame):
        plt.clf()
        img = plt.imread(os.path.join(directory_path, file_list[frame]))
        plt.imshow(img)
        plt.axis('off')
        
    figwidth=9;figheight=7
    fig = plt.figure(figsize=(figwidth,figheight))
    ani = animation.FuncAnimation(fig, animate, frames=len(file_list), interval=200)
    writer = animation.writers['pillow'](fps=1.25)
    animation_name='2D_plot_movie_x_range_'+str(bound)+'_Percentile_'+str(percentile_no)+'%_X'+str(index1)+'vs_X'+str(index2)+'.gif'
    ani.save(animation_name, writer=writer)