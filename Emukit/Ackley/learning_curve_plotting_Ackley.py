import subprocess
import numpy as np
import time
import math
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
import GPy
import csv
import os
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import Norm_Dnorm
import TestFunction
import Final_Model
import record_data_log_file



# Define the negative log-likelihood function for a normal distribution with respect to X=0
def negative_log_likelihood(params, data, x0):
    mean, std_dev = params
    n = len(data)
    log_likelihood = -n/2 * np.log(2*np.pi*std_dev**2) - (1/(2*std_dev**2)) * np.sum((data - (mean))**2)
    
    return -log_likelihood  # We negate the log-likelihood to minimize it

def likelihood(params, data, x0):
    mean, std_dev = params
    n = len(data)
    likelihood = pow((2*np.pi*std_dev**2),-n/2) - (1/(2*std_dev**2)) * np.sum((data - (mean + x0))**2)
    return likelihood  # We negate the log-likelihood to minimize it

def euclidean_distance_hartmann(x,iteration_num):
   if iteration_num==0:
        x0 = ((x[0]) - 0.20169)**2
        x1 = ((x[1]) - 0.150011)**2
        x2 = ((x[2]) - 0.476874)**2
        x3 = ((x[3]) - 0.275332)**2
        x4 = ((x[4]) - 0.311652)**2
        x5 = ((x[5]) - 0.6573)**2
        #print(np.sqrt(x0+x1+x2+x3+x4+x5))
        return np.sqrt(x0+x1+x2+x3+x4+x5)
   else:
        xmin=-1;xmax=1
        x=Norm_Dnorm.normalizer(np.array(x),xmax,xmin)        
        x0 = ((x[0]) - 0.20169)**2
        x1 = ((x[1]) - 0.150011)**2
        x2 = ((x[2]) - 0.476874)**2
        x3 = ((x[3]) - 0.275332)**2
        x4 = ((x[4]) - 0.311652)**2
        x5 = ((x[5]) - 0.6573)**2
        #print(np.sqrt(x0+x1+x2+x3+x4+x5))
        return np.sqrt(x0+x1+x2+x3+x4+x5)

def euclidean_distance_ackley(x,iteration_num):
    xmax=32.768; xmin=-32.768
    #x=Norm_Dnorm.normalizer(np.array(x),xmax,xmin)
    x0 = ((x[0]) - 0.0)**2
    x1 = ((x[1]) - 0.0)**2
    x2 = ((x[2]) - 0.0)**2
    x3 = ((x[3]) - 0.0)**2
    x4 = ((x[4]) - 0.0)**2
    x5 = ((x[5]) - 0.0)**2
#print(np.sqrt(x0+x1+x2+x3+x4+x5))
    return np.sqrt(x0+x1+x2+x3+x4+x5)

def saveing_in_folder(folder_name,plot_filename):
    folder_path = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    plt.savefig(plot_filepath,transparent='clear',facecolor='white',bbox_inches='tight')
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

def scatter_25_and_75_percentile_track_X(Output,n_lhs_samples,total,bs,iterations,ground_y_max,beta,std_noise,filename,ker_variance_filename):
    path=os.getcwd() 
    ground_x_max = 0    
    output = []

    #data reading .................................
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        ctr = 0
        for row in reader:
            
            if len(row) == 1: 
                           
               if row[0] == 100.0:
                print('Break done')
                break;
               ctr += 1
               output.append([])  
                    
            else:
               output[ctr-1].append((row[0], row[1:]))

    total_iter = len(output[0])
    
    # initialization variables   
    medians_Y = 0
    medians_X=  0    
    percentile25 = 0
    percentile75 = 0
    percentile25_X =0
    percentile75_X = 0
    percentile25_X_index = 0
    percentile75_X_index = 0
    medians_X_index=0
    index_Y= 0
    ground_x_max=0
    num_of_iterations=len(output[0])

    
    #fig features 
    figwidth=2.5
    figheight=2
    s_value=7
    s_value_big=10
    fig1=plt.figure(figsize=(figwidth,figheight))
    plt.style.use('seaborn-paper')

    # X learning curve plotting start ...........
    for iteration_num in range(len(output[0])):
        data = [euclidean_distance_ackley(run[iteration_num][1],iteration_num) for run in output] 
        data=np.array(data);x=iteration_num;length=len(data)
        plt.scatter(np.repeat(x,length), data ,s=s_value,color='#6A6B72',marker='*') 
    plt.scatter(np.repeat(x,length), data, s=s_value ,color='#6A6B72',marker='*',label='$\mathbf{X}$ for 99 LHS')
    
    
    # # Taking percentile of 25, 50 and 75 of X and plotting that    
    medians_X=np.median(data)
    indexY=0  
    for d in data:
        if medians_X==d:
            break;
        indexY+=1 

    #25%     
    sorted_data=np.sort(data)        
    index_25th_percentile = int(0.75 * len(data))
    value_25th_percentile = sorted_data[index_25th_percentile]
    index25=0
    for d in data:            
        if value_25th_percentile==d:
            break;
        index25+=1   

    label='25% Percentile'     
    for iteration_num in range(len(output[0])): 
        dataY25=euclidean_distance_ackley(output[index25][iteration_num][1],iteration_num)      
        plt.scatter(iteration_num,dataY25,s=s_value_big,marker='v',facecolors='#44F65C')
    plt.scatter(iteration_num-1,dataY25,s=s_value_big,marker='v',facecolors='#44F65C',label=label)

    # 75%     
    index_75th_percentile = int(0.25 * len(data))
    value_75th_percentile = sorted_data[index_75th_percentile]
    index75=0
    for d in data:            
        if value_75th_percentile==d:
            break;
        index75+=1 

    label='50% Percentile'
    data50=[];ACR_X=0
    for iteration_num in range(len(output[0])):  
        dataY=euclidean_distance_ackley(output[indexY][iteration_num][1],iteration_num)   
        data50.append(dataY) 
        plt.scatter(iteration_num,dataY,s=s_value_big,color='red',marker='o')
    plt.scatter(iteration_num,dataY,s=s_value_big,color='red',marker='o',label=label)
    plt.step(range(0,len(output[0]),1),data50,linewidth=0.35,color='red')


    label='75% Percentile'     
    for iteration_num in range(len(output[0])): 
        dataY75=euclidean_distance_ackley(output[index75][iteration_num][1],iteration_num)       
        plt.scatter(iteration_num,dataY75, s=s_value_big,marker='^',facecolors='#152CEB')
    plt.scatter(iteration_num-1,dataY75, s=s_value_big,marker='^',facecolors='#152CEB',label=label)

    
    # end...................................
    plt.axhline(y = ground_x_max, color = '#C49A17', linestyle = 'dashed',label='Ground Truth Max')
    xticks=[0,10,20,30,40,50]
    yticks=[0,10,20,30,40,50]
    plt.xticks(xticks)
    plt.yticks(yticks)  
    plt.tick_params(direction='in') 
    plt.xlabel("Iteration")
    plt.ylabel("||$\mathbf{X}$ - $\mathbf{X}_{\mathrm{max}}$||")
    #plt.ylim(-3.5,50)
    #plt.xlim(-1,51)
    # legend = plt.legend()
    # legend.get_frame().set_alpha(0.6)
    # plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()
    plt.savefig('Learning_curve_X_scatter_track_X.svg')    
    plt.close()

    #y learning curve..........................  
    fig1=plt.figure(figsize=(figwidth,figheight))
    plt.style.use('seaborn-paper')      
    num_of_iterations=len(output[0])  

    for iteration_num in range(len(output[0])):
        data = [run[iteration_num][0] for run in output] 
        data=np.array(data);x=iteration_num;length=len(data)       
        plt.scatter(np.repeat(x,length), data ,s=s_value,color='#6A6B72',marker='*') 
    plt.scatter(np.repeat(x,length), data ,s=s_value,color='#6A6B72',marker='*',label='y for 99 LHS') 

    
    label='25% Percentile'
    for iteration_num in range(len(output[0])): 
        dataX25=output[index25][iteration_num][0]    
        plt.scatter(iteration_num,dataX25,s=s_value_big, marker='v',facecolors='#44F65C')
    plt.scatter(iteration_num-1,dataX25,s=s_value_big, marker='v',facecolors='#44F65C',label=label)

    label='50% Percentile'
    data50=[];
    for iteration_num in range(len(output[0])):  
        dataX=output[indexY][iteration_num][0]
        data50.append(dataX)     
        plt.scatter(iteration_num,dataX,s=s_value_big,color='red',marker='o')
    plt.scatter(iteration_num-1,dataX,s=s_value_big,color='red',marker='o',label=label)
    plt.step(range(0,len(output[0]),1),data50,linewidth=0.35,color='red')

    label='75% Percentile'
    for iteration_num in range(len(output[0])): 
        dataX75=output[index75][iteration_num][0]      
        plt.scatter(iteration_num,dataX75,s=s_value_big, marker='^',facecolors='#152CEB')
    plt.scatter(iteration_num-1,dataX75,s=s_value_big, marker='^',facecolors='#152CEB',label=label)

    
    # end..............
        
    plt.axhline(y = ground_y_max, color = '#C49A17', linestyle = 'dashed',label='Ground Truth Max')
    xticks=[0,10,20,30,40,50]
    yticks=[-20,-15,-10,-5,0]
    plt.xticks(xticks)
    plt.yticks(yticks)  
    plt.tick_params(direction='in') 
    plt.xlabel("Iteration")
    plt.ylabel("Max(μ_D($\mathbf{X}$))")
    #plt.ylim(-22,2.5)
    #plt.xlim(-1,51)
    plt.tight_layout()
    plt.savefig('Learning_curve_Y_scatter_track_X.svg')
    plt.close()

   

    sum_IR_X=0
    sum_IR_Y=0

    for run in output:
        data = euclidean_distance_ackley(run[len(output[0])-1][1],iteration_num=49) 
        sum_IR_X+=data
        data = np.abs(ground_y_max-run[len(output[0])-1][0]) 
        sum_IR_Y+=data

    sum_CR_X=0
    sum_CR_Y=0
    total_iteration=int(len(output[0]))
    for run in output:
        sum_X=0;
        sum_Y=0
        for i in range(total_iteration):
            data = euclidean_distance_ackley(run[i][1],iteration_num=i) 
            sum_X+=data
            data = np.abs(ground_y_max-run[i][0]) 
            sum_Y+=data
        sum_CR_X+=sum_X
        sum_CR_Y+=sum_Y
    sum_CR_X/=99
    sum_CR_Y/=99

    sum_IR_X/=99
    sum_IR_Y/=99


    file_name='LOG_TABLE.csv'
    record_data_log_file.log_table(beta,std_noise,sum_IR_Y,sum_CR_Y,sum_IR_X,sum_CR_X,file_name)
    return indexY,index25,index75;

 
def kernel_parameter_lengthscale_plotting(Output,n_lhs_samples,total,bs,iterations,ground_y_max,beta,index,percentile25_X_index,percentile75_X_index,filename):
    path=os.getcwd()
    ground_x_max = 0  
    
    # data reading.............
    output = []
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        ctr = 0
        for row in reader:
            if len(row) == 1:
               if row[0] == 100.0:
                print('Break done')
                break;
               ctr += 1
               output.append([])
            else:               
               output[ctr-1].append((row[0], row[1],row[2],row[3],row[4],row[5]))

    # plotting
    nvar=6;
    last_lengthscale=[]
    for i in range(nvar):
        # initialization for each lengthscale
        medians = 0
        percentile75 = 0
        percentile25 = 0        
        max_likelihood=0
        max_log_likelihood=0
        data_Lengthscale=0
        num_of_iterations=len(output[0])

        #figure features
        s_value=7
        s_value_big=10
        figwidth=2.5;figheight=2
        fig1=plt.figure(figsize=(figwidth,figheight))
        plt.style.use('seaborn-paper') 
        
        for iteration_num in range(len(output[0])):
            data = [run[iteration_num][i] for run in output]
            x=iteration_num
            length=len(data)
            plt.scatter(np.repeat(x,length), data ,s=s_value,color='#6A6B72',marker='*') 
        plt.scatter(np.repeat(x,length), data ,s=s_value,color='#6A6B72',marker='*',label='99 LHS')

        label='25% Percentile' 
        for iteration_num in range(len(output[0])): 
            data=output[percentile25_X_index][iteration_num][i]      
            plt.scatter(iteration_num,data,s=s_value_big, marker='v',facecolors='#44F65C')
        plt.scatter(iteration_num-1,data,s=s_value_big, marker='v',facecolors='#44F65C',label=label)

        label='50% Percentile'
        data50=[]
        for iteration_num in range(len(output[0])): 
            dataY=output[index][iteration_num][i]     
            data50.append(dataY) 
            plt.scatter(iteration_num,dataY,s=s_value_big,color='red',marker='o')
        plt.scatter(iteration_num,dataY,s=s_value_big,color='red',marker='o',label=label)
        plt.step(range(0,len(output[0]),1),data50,linewidth=0.35,color='red')
        last_data=dataY

        label='75% Percentile' 
        for iteration_num in range(len(output[0])): 
            data=output[percentile75_X_index][iteration_num][i]      
            plt.scatter(iteration_num,data,s=s_value_big, marker='^',facecolors='#152CEB')
        plt.scatter(iteration_num-1,data,s=s_value_big, marker='^',facecolors='#152CEB',label=label)
       
        
        label1=str(last_data) 
        label='Lengthscale '+str(i+1)
        yticks=[0.0,0.2,0.4,0.6,0.8,1.0]
        xticks=[0,10,20,30,40,50]
        plt.xticks(xticks)
        plt.yticks(yticks)  
        plt.tick_params(direction='in')      
        plt.xlabel("Iteration")
        plt.ylabel(label)
        plt.xlim(-1,51)

        #plt.legend(prop=legend_font,loc='upper right', bbox_to_anchor=(0.9, 0.7))
        fig_name='Kernel_Lengthscale_'+str(i+1)+'_beta_'+str(beta)+'.svg'
        plt.tight_layout()
        plt.savefig(fig_name)
        plt.close() 

        last_lengthscale.append(last_data)
    return np.array(last_lengthscale) 


def kernel_variance_plotting(Output,n_lhs_samples,total,bs,iterations,ground_y_max,beta,index,percentile25_X_index,percentile75_X_index,filename):
    
    path=os.getcwd();ground_x_max = 0    
    #data reading
    output = []
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        ctr = 0
        for row in reader:
            if len(row) == 1:
               ctr += 1
               output.append([])
            else:
               output[ctr-1].append((row[0], row[1]))

    total_iter = len(output[0]) 
    medians = 0
    mean_=0
    percentile75 = 0
    percentile25 = 0
    
    
    s_value=7
    s_value_big=10
    figwidth=2.5;figheight=2
    fig1=plt.figure(figsize=(figwidth,figheight))
    plt.style.use('seaborn-paper')
     
    for iteration_num in range(len(output[0])):
        data = [run[iteration_num][0] for run in output]  
        data=np.sqrt(data)  
        x=iteration_num
        length=len(data)
        plt.scatter(np.repeat(x,length), data ,s=s_value,color='#6A6B72',marker='*') 
    plt.scatter(np.repeat(x,length), data ,s=s_value,color='#6A6B72',marker='*',label='99 LHS')

    label='25% Percentile' 
    for iteration_num in range(len(output[0])): 
        data=output[percentile25_X_index][iteration_num][0] 
        data=np.sqrt(data)     
        plt.scatter(iteration_num,data,s=s_value_big, marker='v',facecolors='#44F65C')
    plt.scatter(iteration_num-1,data,s=s_value_big, marker='v',facecolors='#44F65C',label=label)

    
    label='50% Percentile'
    data50=[]
    for iteration_num in range(len(output[0])): 
        dataY=output[index][iteration_num][0] 
        dataY=np.sqrt(dataY)    
        data50.append(dataY) 
        plt.scatter(iteration_num,dataY,s=s_value_big,color='red',marker='o')
    plt.scatter(iteration_num,dataY,s=s_value_big,color='red',marker='o',label=label)
    plt.step(range(0,len(output[0]),1),data50,linewidth=0.35,color='red')
    last_data=dataY

    label='75% Percentile' 
    for iteration_num in range(len(output[0])): 
        data=output[percentile75_X_index][iteration_num][0]
        data=np.sqrt(data)      
        plt.scatter(iteration_num,data,s=s_value_big, marker='^',facecolors='#152CEB')
    plt.scatter(iteration_num-1,data,s=s_value_big, marker='^',facecolors='#152CEB',label=label)
    

    
    label1=str(last_data) 
    label='Kernel Amplitude'
    yticks=[0.0,0.1,0.2,0.3]
    xticks=[0,10,20,30,40,50]
    plt.xticks(xticks)
    plt.yticks(yticks)  
    plt.tick_params(direction='in')      
    plt.xlabel("Iteration")
    plt.ylabel(label)
    #plt.ylim(-0.02,50)
    plt.xlim(-1,51)
    
    figname='Kernel_Amplitude'+'_beta_'+str(beta)+'.svg'
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()

    return last_data

def gaussian_variance_plotting(Output,n_lhs_samples,total,bs,iterations,ground_y_max,beta,index,percentile25_X_index,percentile75_X_index,filename):
    
    path=os.getcwd() 
    ground_x_max = 0    
    output = []

    
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        ctr = 0
        for row in reader:
            if len(row) == 1:
               ctr += 1
               output.append([])
            else:
               output[ctr-1].append((row[0], row[1]))

    total_iter = len(output[0]) 
    
    medians = 0
    mean_=0
    percentile75 = 0
    percentile25 = 0
    s_value=7
    s_value_big=10
    figwidth=2.5;figheight=2
    fig1=plt.figure(figsize=(figwidth,figheight))
    plt.style.use('seaborn-paper')
     
    for iteration_num in range(len(output[0])):
        data = [run[iteration_num][1] for run in output] 
        x=iteration_num
        length=len(data)
        plt.scatter(np.repeat(x,length), data ,s=s_value,color='#6A6B72',marker='*') 
    plt.scatter(np.repeat(x,length), data ,s=s_value,color='#6A6B72',marker='*',label='99 LHS')

    label='25% Percentile' 
    for iteration_num in range(len(output[0])): 
        data=output[percentile25_X_index][iteration_num][1]      
        plt.scatter(iteration_num,data,s=s_value_big, marker='v',facecolors='#44F65C')
    plt.scatter(iteration_num-1,data,s=s_value_big, marker='v',facecolors='#44F65C',label=label)

    
    label='50% Percentile'
    data50=[]
    for iteration_num in range(len(output[0])): 
        dataY=output[index][iteration_num][1]     
        data50.append(dataY) 
        plt.scatter(iteration_num,dataY,s=s_value_big,color='red',marker='o')
    plt.scatter(iteration_num,dataY,s=s_value_big,color='red',marker='o',label=label)
    plt.step(range(0,len(output[0]),1),data50,color='red')
    last_data=dataY

    label='75% Percentile' 
    for iteration_num in range(len(output[0])): 
        data=output[percentile75_X_index][iteration_num][1]      
        plt.scatter(iteration_num,data,s=s_value_big, marker='^',facecolors='#152CEB')
    plt.scatter(iteration_num-1,data,s=s_value_big, marker='^',facecolors='#152CEB',label=label)
    

    
    label1=str(last_data) 
    label='Gaussian Variance '  
    
    xticks=[0,10,20,30,40,50]
    plt.xticks(xticks)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    plt.tick_params(direction='in')      
    plt.xlabel("Iteration")
    plt.ylabel(r'GNV ($\times 10^{-2}$)')
    plt.xlim(-1,51)
    yticks=[0.0,0.5,1.0,1.5]
    plt.yticks(yticks)
    plt.ylim(-0,1.5)
    
    figname='Gaussian_Variance'+'_beta_'+str(beta)+'.svg'
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()

    return last_data

def plotting(Output,n_lhs_samples,total,bs,iterations,ground_y_max,beta,std_noise,filename,ker_variance_filename,ker_lengthscale_filename):
    
    # plotting of X and y learning curve
    index_corresponded_Y,percentile25_X_index,percentile75_X_index=scatter_25_and_75_percentile_track_X(Output,n_lhs_samples,total,bs,iterations,ground_y_max,beta,std_noise,filename,ker_variance_filename)
    
    # plotting of hyperparameter evolution curve
    directory_path=os.getcwd()
    pattern = ker_lengthscale_filename
    matching_files = [filename for filename in os.listdir(directory_path) if filename.startswith(pattern)]
    file_lengthscale=matching_files[0]
    ker_lengthscale=kernel_parameter_lengthscale_plotting(Output,n_lhs_samples,total,bs,iterations,ground_y_max,beta,index_corresponded_Y,percentile25_X_index,percentile75_X_index,file_lengthscale)

    directory_path=os.getcwd()
    pattern = ker_variance_filename
    matching_files = [filename for filename in os.listdir(directory_path) if filename.startswith(pattern)]
    file_variance=matching_files[0]
    ker_variance=kernel_variance_plotting(Output,n_lhs_samples,total,bs,iterations,ground_y_max,beta,index_corresponded_Y,percentile25_X_index,percentile75_X_index,file_variance)
    gauss_variance=gaussian_variance_plotting(Output,n_lhs_samples,total,bs,iterations,ground_y_max,beta,index_corresponded_Y,percentile25_X_index,percentile75_X_index,file_variance)

  