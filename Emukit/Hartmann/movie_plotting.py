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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import record_data_log_file
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap

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
    return X1, X2, X3, X4, X5, X6

def model_prediction_per_iteration(model,ymax,ymin,xmax,xmin,frame):

    n= 11
    X1,X2,X3,X4,X5,X6=meshgrid(n)    
    X_ = np.column_stack([X1.ravel(), X2.ravel(), X3.ravel(), X4.ravel(), X5.ravel(), X6.ravel()])


    # Compute y_pred using the GP model
    f_obj=model[frame]
    y_pred, ___ = f_obj(X_); 
    y_pred=-Norm_Dnorm.denormalizer(y_pred, ymax, ymin);y_pred = np.array(y_pred)       
    X = np.array(X_)

    idx = np.argmax(y_pred)
    X_=Norm_Dnorm.denormalizer(X_, xmax, xmin);
    X1max, X2max, X3max, X4max, X5max, X6max = X_[idx, 0], X_[idx, 1], X_[idx, 2], X_[idx, 3], X_[idx, 4], X_[idx, 5]
    X1true, X2true, X3true, X4true, X5true, X6true = 0.20169, 0.150011, 0.476874, 0.275322, 0.311652, 0.65730
    X1true1, X2true1, X3true1, X4true1, X5true1, X6true1 = 0.40460235,0.88231846,0.81636135,0.57388939,0.14649013,0.03864136


    #predicted model for corresponding X  variable

    Ycube = np.reshape(y_pred, (n, n, n, n, n, n))
    M12 = np.max(Ycube, axis=(2, 3, 4, 5));M12 = np.squeeze(M12)
    M13 = np.max(Ycube, axis=(1, 3, 4, 5));M13 = np.squeeze(M13)
    M14 = np.max(Ycube, axis=(1, 2, 4, 5));M14 = np.squeeze(M14)
    M15 = np.max(Ycube, axis=(1, 2, 3, 5));M15 = np.squeeze(M15)
    M16 = np.max(Ycube, axis=(1, 2, 3, 4));M16 = np.squeeze(M16)

    return M12,M13,M14,M15,M16,X1max,X2max,X3max,X4max,X5max,X6max,X1true,X2true,X3true,X4true,X5true,X6true,X1true1, X2true1, X3true1, X4true1, X5true1, X6true1;
def update(frame,X1,X2,f_obj,X,xmax,xmin,ymax,ymin,bound,frame_length,index1,index2,ax):

    #prediction of corresping X variables model per iteration
    M12,M13,M14,M15,M16,X1max,X2max,X3max,X4max,X5max,X6max,X1true,X2true,X3true,X4true,X5true,X6true,X1true1, X2true1, X3true1, X4true1, X5true1, X6true1=model_prediction_per_iteration(f_obj,ymax,ymin,xmax,xmin,frame)
    
    
    cmap='plasma'
    if index1==1 and index2==2:
        cbar=ax.plot_surface(X2, X1, M12,cmap=cmap,zorder=1)
    elif index1==1 and index2==3:
        cbar=ax.plot_surface(X2, X1, M13,cmap=cmap,zorder=1)
    elif index1==1 and index2==4:
        cbar=ax.plot_surface(X2, X1, M14,cmap=cmap,zorder=1)
    elif index1==1 and index2==5:
        cbar=ax.plot_surface(X2, X1, M15,cmap=cmap,zorder=1)
    elif index1==1 and index2==6:
        cbar=ax.plot_surface(X2, X1, M16,cmap=cmap,zorder=1)
    
    xyticks=[0.0,0.5,1.0]
    zticks=[0,1,2,3]
    plt.xlabel('x'+str(index1), labelpad=0.1);plt.xticks(xyticks)
    plt.ylabel('x'+str(index2), labelpad=0.1);plt.yticks(xyticks)
    ax.set_zticks(zticks) 
    plt.tick_params(direction='in',pad=0.1) 
    
    #BO points projection per iteration
    
    # Create a segmented color map with varying segment lengths
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
    }
    positions = [0.0, 0.33, 0.67, 1.0]

    D=[70, 254, 75  ]
    C=[70, 254, 196    ]
    B=[70, 191, 254   ]
    A=[ 70, 85, 254 ]

    # Define colors and their corresponding positions for transition
    colors = [(A[0]/255,A[1]/255,A[2]/255), (B[0]/255,B[1]/255,B[2]/255), (C[0]/255,C[1]/255,C[2]/255), (D[0]/255,D[1]/255,D[2]/255)]
    for pos, color in zip(positions, colors):
        r, g, b = color
        cdict['red'].append((pos, r, r))
        cdict['green'].append((pos, g, g))
        cdict['blue'].append((pos, b, b))

    custom_cmap = LinearSegmentedColormap('CustomRed', cdict)
    cmap=custom_cmap   
    
    # Bo color mapping
    normalize=plt.Normalize(0,frame_length)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    sm.set_array([])  # You can set this to a specific array if needed


    #LHS points projection     
    n_lhs=24;
    marker_size=20
    height_for_visibility=0.0
    Y_truth=TestFunction.hartmann(X)
    for i in range(n_lhs):
        x=X[i]
        y_truth=Y_truth[i]
        y_truth+=height_for_visibility
        ax.scatter(x[index1-1],x[index2-1],y_truth,s=marker_size, zorder=3, facecolor='red',edgecolors='none',marker='o')
    ax.scatter(x[index1-1],x[index2-1],y_truth,s=marker_size, zorder=3, facecolor='red',edgecolors='none',marker='o',label='LHS Points')

    # model points projection
    plt.title(f'Iteration {frame+1}')
    index_start=24;index_end=int(24+(frame)*4)+4
    color=cmap(normalize(frame))
    ctr=0
    for i in range(index_start,index_end,1): 
        temp=(i)%4
        if temp==0:
            index=int(float(i/4))
            color=cmap(normalize(index))
        x=X[i]  
        y_truth=Y_truth[i]
        ax.scatter(x[index1-1],x[index2-1],y_truth,s=marker_size, zorder=2, facecolor=[color],edgecolor='none')
    ax.scatter(x[index1-1],x[index2-1],y_truth,s=marker_size, zorder=2, facecolor=[color],edgecolors='none',marker='o',label='BO Points')
    gt_marker_size=80
    xtruez1=3.32237; xtruez2=3.22231
    if index1==1 and index2==2:
        ax.scatter(X1true, X2true,xtruez1, s=gt_marker_size-30, zorder=3, color='black', marker='x', label='GT Max')
        ax.scatter(X1true1, X2true1,xtruez2, s=gt_marker_size, zorder=3, color='#C11B17', marker='*', label='2nd True Max')

    elif index1==1 and index2==3:
        ax.scatter(X1true, X3true,xtruez1, s=gt_marker_size-30, zorder=3, color='black', marker='x', label='GT Max')
        ax.scatter(X1true1, X3true1,xtruez2, s=gt_marker_size, zorder=3, color='#C11B17', marker='*', label='2nd True Max')

    elif index1==1 and index2==4:
        ax.scatter(X1true, X4true,xtruez1, s=gt_marker_size-30, zorder=3, color='black', marker='x', label='GT Max')
        ax.scatter(X1true1, X4true1,xtruez2, s=gt_marker_size, zorder=3, color='#C11B17', marker='*', label='2nd True Max')

    elif index1==1 and index2==5:
        ax.scatter(X1true, X5true,xtruez1, s=gt_marker_size-30, zorder=3, color='black', marker='x', label='GT Max')
        ax.scatter(X1true1, X5true1,xtruez2, s=gt_marker_size, zorder=3, color='#C11B17', marker='*', label='2nd True Max')

    elif index1==1 and index2==6:
        ax.scatter(X1true, X6true,xtruez1, s=gt_marker_size-30, zorder=3, color='black', marker='x', label='GT Max')
        ax.scatter(X1true1, X6true1,xtruez2, s=gt_marker_size, zorder=3, color='#C11B17', marker='*', label='2nd True Max')

    
                    
    print('Frame No==',frame)
    plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.5)

def movie_plot(beta, x1, x2, f_obj, model_X,X_new_dn,xmax, xmin, X1true, X2true,ymax,ymin,bound,percentile_no,std_noise,index1,index2):

    figwidth=2.2;figheight=2;
    fig, ax = plt.subplots(figsize=(figwidth,figheight))
    X1, X2 = np.meshgrid(x1, x2)
    xmax=1;xmin=0

          
    figwidth=2.2
    figheight=2

    frame_length=int((len(model_X)-24)/4) 
    folder_name='Movie_plot_'+'_Percentile_'+str(percentile_no)+'%_X'+str(index1)+'_X'+str(index2)    
    for frame_number in range(frame_length):
        temp=int((frame_number+1)%4)
        if frame_number<=20 or frame_number==(frame_length-1) or frame_number==22 or frame_number==27 or frame_number==32:
            fig1 = plt.figure(figsize=(figwidth,figheight))
            ax=fig1.add_subplot(1, 1, 1,projection='3d',computed_zorder=False)
            plt.style.use('seaborn-paper')
            update(frame_number, X1, X2, f_obj, model_X, xmax, xmin, ymax, ymin, bound, frame_length,index1,index2,ax)                       
            plot_filename='frame_'+str(frame_number)+'.png'
            plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.5)
            saveing_in_folder(folder_name,plot_filename)
            plt.close()
        elif frame_number>20 and temp==0:
            fig1 = plt.figure(figsize=(figwidth,figheight))
            ax=fig1.add_subplot(1, 1, 1,projection='3d',computed_zorder=False)
            plt.style.use('seaborn-paper')
            update(frame_number, X1, X2, f_obj, model_X, xmax, xmin, ymax, ymin, bound, frame_length,index1,index2,ax)
            plot_filename='frame_'+str(frame_number)+'.png'
            plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.5)
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
    
    # Set up the figure and create the animation
    fig = plt.figure(figsize=(9,7))
    ax = plt.axes()
    ax.set_facecolor('black')
    ani = animation.FuncAnimation(fig, animate, frames=len(file_list), interval=200)
    writer = animation.writers['pillow'](fps=1.25)
    animation_name='2D_plot_movie_beta'+str(beta)+'_Noise_'+str(std_noise)+'_Percentile_'+str(percentile_no)+'%_X'+str(index1)+'vs_X'+str(index2)+'.gif'
    folder_path = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    ani_filepath = os.path.join(folder_path, animation_name)    
    ani.save(ani_filepath, writer=writer)

