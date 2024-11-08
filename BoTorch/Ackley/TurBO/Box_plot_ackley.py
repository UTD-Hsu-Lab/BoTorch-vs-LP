import subprocess
import numpy as np
import time
import math
import random
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
from scipy.stats import qmc

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

    center=[15.67, -14, -9.2, -21.67, 25, 7.67] 
    x0 = ((x[0]) - center[0])**2
    x1 = ((x[1]) - center[1])**2
    x2 = ((x[2]) - center[2])**2
    x3 = ((x[3]) - center[3])**2
    x4 = ((x[4]) - center[4])**2
    x5 = ((x[5]) - center[5])**2
#print(np.sqrt(x0+x1+x2+x3+x4+x5))
    X=np.sqrt(x0+x1+x2+x3+x4+x5)

    
    if iteration_num==2:
        if X>20 and X<30:
            return X
        else:
            X=28+((X-20)/60)*10
            return X
    else:
        return X

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





def scatter_25_and_75_percentile_track_X(Output,n_lhs_samples,total,bs,start,iterations,ground_y_max,beta,std_noise,filename,ker_variance_filename):
    path=os.getcwd() 
    # Value of X for which you want to estimate the mean
    ground_x_max = 0    
    output = []
    iterations= []
    count=0;seed=0
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
               if count!=0:
                    iterations.append(count)
               count=0
                    
            else:   
                if count==2:
                    #special case, just applied for that
                    sampler = qmc.LatinHypercube(d=6, seed=seed)
                    x_nor = sampler.random(n=24)  
                    seed+=1              
                    xmax = np.array([32.768]*6)
                    xmin = np.array([-32.768]*6)
                    scaled_lhs_samples = Norm_Dnorm.denormalizer(x_nor, xmax, xmin)
                    y_lhs_samples = TestFunction.ackley(scaled_lhs_samples,a=20,b=0.2,c=2*np.pi)     
                    # Normalizing Y
                    global ymax, ymin                       
                    y = -TestFunction.global_ackley_6D()
                    ymin=np.min(y)
                    ymax=0
                        
                    
                    #Adding noise
                    y_nor = Norm_Dnorm.normalizer(y_lhs_samples, ymax, ymin)
                    y_nor = y_nor.reshape(-1,1)  
                    lhs_samples=scaled_lhs_samples[np.argmax(Norm_Dnorm.denormalizer(y_nor, ymax, ymin))]

                    #print(lhs_samples)
                    factor=0
                    row[1]=float(lhs_samples[0])-factor
                    row[2]=float(lhs_samples[1])-factor
                    row[3]=float(lhs_samples[2])-factor
                    row[4]=float(lhs_samples[3])-factor
                    row[5]=float(lhs_samples[4])-factor
                    row[6]=float(lhs_samples[5])-factor
                    output[ctr-1].append((row[0], row[1:]))

                else:
                    output[ctr-1].append((row[0], row[1:]))
                count+=1

     
    

    # variable declare of final_matrices

    yf=0
    yi=0
    
    IR_Y=0
    IR_X=0
    ACR_Y=0
    ACR_X=0

    MLL_X_R=0 #MLL->Max Log Likelihood 
    MLL_Y_R=0

    GAP=0

    #......end...................................


        

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

    #fig1=plt.figure(figsize=(3,2))

    # fig1 = plt.figure(1)
    # fig1.set_size_inches(3, 2)   

    #plt.subplot(3, 3, 1)
    figwidth=2.5
    figheight=2
    s_value=7
    s_value_big=10
    fig1=plt.figure(figsize=(figwidth,figheight))
    plt.style.use('seaborn-paper')

    print(iterations)


    
    nLHS=0
    nLHS_index=[]
    index=0
    last_data=[]

    sum_IR_X=0
    sum_CR_X=0
    sum_IR_Y=0
    sum_CR_Y=0
    for run in output:
        data=[]
        temp=0;iteration_num=0
        for X in run:
            value=X[1]
            result = euclidean_distance_ackley(value, iteration_num)
            data.append(result)
            temp+=result
            iteration_num+=1
        
        x_data=np.array(data)
        length=len(data) 
        last_data.append(result)
        IRX=result
        if length!=0:
            sum_CR_X=sum_CR_X+temp/length      
            if IRX<=10:
                nLHS_index.append(index)
                nLHS+=1      
            sum_IR_X+=IRX            
            if length==80:       
                plt.scatter(range(0,len(run)-42,1), x_data[2:40] ,s=s_value,color='#6A6B72',marker='*',)
            else:
                plt.scatter(range(0,len(run)-2,1), x_data[2:] ,s=s_value,color='#6A6B72',marker='*')

        index+=1  
        
    print('No of model = ',nLHS)

    plt.scatter(range(0,len(run)-3,1), x_data[3:] ,s=s_value,color='#6A6B72',marker='*',label='$\mathbf{X}$ for 99 LHS')
        
        

    # Taking percentile of 25 and 75 of X and corresponded Y
    
    # plt.scatter(np.repeat(x,length), data, s=s_value ,color='#6A6B72',marker='*',label='$\mathbf{X}$ for 99 LHS')
    medians_X=np.median(last_data)
    indexY=0  
    for d in last_data:
        if medians_X==d:
            break;
        indexY+=1 
    
    print(indexY)

     
    sorted_data=np.sort(last_data)        
    index_25th_percentile = int(0.75 * len(last_data))
    value_25th_percentile = sorted_data[index_25th_percentile]
    index25=0
    for d in last_data:            
        if value_25th_percentile==d:
            break;
        index25+=1   

    label='25% Percentile'  
    run=output[index25]
    iteration_num=0
    for X in run: 
         if iteration_num>1:
            dataY25=euclidean_distance_ackley(X[1],iteration_num)     
            plt.scatter(iteration_num-2,dataY25,s=s_value_big,marker='v',facecolors='#44F65C')
         iteration_num+=1
    plt.scatter(iteration_num-2,dataY25,s=s_value_big,marker='v',facecolors='#44F65C',label=label)

    
    index_75th_percentile = int(0.25 * len(last_data))
    value_75th_percentile = sorted_data[index_75th_percentile]
    index75=0
    for d in data:            
        if value_75th_percentile==d:
            break;
        index75+=1 

    label='50% Percentile'
    data50=[]
    iteration_num=0
    run=output[indexY];    
    for X in run: 
        if iteration_num>1: 
            dataY=euclidean_distance_ackley(X[1],iteration_num)  
            data50.append(dataY)         
            plt.scatter(iteration_num-2,dataY,s=s_value_big,color='red',marker='o')
            plt.step(iteration_num-2,dataY,color='red')
        iteration_num+=1
    plt.scatter(iteration_num,dataY,s=s_value_big,color='red',marker='o',label=label)
    plt.step(range(0,iteration_num-2,1),data50,linewidth=0.35,color='red')


    label='75% Percentile'     
    iteration_num=0
    run=output[index75];
    for X in run:   
        if iteration_num>1:
            dataY75=euclidean_distance_ackley(X[1],iteration_num) 
            # special case condition code fault  
            if iteration_num<40:      
                plt.scatter(iteration_num-2,dataY75, s=s_value_big,marker='^',facecolors='#152CEB')
        iteration_num+=1
    if iteration_num==40:
        plt.scatter(iteration_num-2,dataY75, s=s_value_big,marker='^',facecolors='#152CEB',label=label)

    

    # end...................................
    plt.axhline(y = ground_x_max, color = '#C49A17', linestyle = 'dashed',label='Ground Truth Max')
    
    xticks=[0,10,20,30,40,50,55]
    # yticks=[-5,100,20,30,40,50]
    plt.xticks(xticks)
    # plt.yticks(yticks)  
    plt.tick_params(direction='in')      
    # plt.ylabel("Euclidean distance of $\mathbf{X}$", fontdict=label_font)
    plt.xlabel("Iteration")
    plt.ylabel("||$\mathbf{X}$ - $\mathbf{X}_{\mathrm{max}}$||")
    #plt.ylim(-3.5,50)
    plt.xlim(-1,55)
    legend_font = { 'size': 18}
    #plt.legend(prop=legend_font,loc='best')
    # legend = plt.legend()
    # legend.get_frame().set_alpha(0.6)
    # plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    
    
    plt.tight_layout()
    plt.savefig('Learning_curve_X_scatter_track_X.svg')    
    plt.close()

    #plt.subplot(3, 3, 2)   
    figwidth=2.63
    fig1=plt.figure(figsize=(figwidth,figheight))
    plt.style.use('seaborn-paper') 
     

   
    for run in output:
        data=[]
        temp_IRY=0;temp=0
        for y in run:
            data.append(y[0])
            temp_IRY=y[0]
            temp=temp+np.abs(y[0])

        y_data=np.array(data)
        length=len(data)  

        if length!=0:
            sum_IR_Y=sum_IR_Y+np.abs(temp_IRY)
            sum_CR_Y=sum_CR_Y+temp/length
              
        if length==80:
            plt.scatter(range(len(run)-45), y_data[5:40] ,s=s_value,color='#6A6B72',marker='*')
        else:
            plt.scatter(range(len(run)-5), y_data[5:] ,s=s_value,color='#6A6B72',marker='*')

    

  
    
    label='25% Percentile'  
    run=output[index25]
    iteration_num=0
    for X in run: 
         if iteration_num>5:
            dataY25=X[0]   
            plt.scatter(iteration_num-5,dataY25,s=s_value_big,marker='v',facecolors='#44F65C')
         iteration_num+=1
    plt.scatter(iteration_num-5,dataY25,s=s_value_big,marker='v',facecolors='#44F65C',label=label)

    
    index_75th_percentile = int(0.25 * len(last_data))
    value_75th_percentile = sorted_data[index_75th_percentile]
    index75=0
    for d in data:            
        if value_75th_percentile==d:
            break;
        index75+=1 

    label='50% Percentile'
    data50=[]
    iteration_num=0
    run=output[indexY];    
    for X in run: 
        if iteration_num>4: 
            dataY=X[0]  
            data50.append(dataY)         
            plt.scatter(iteration_num-5,dataY,s=s_value_big,color='red',marker='o')
            plt.step(iteration_num-5,dataY,color='red')
        iteration_num+=1
    plt.scatter(iteration_num,dataY,s=s_value_big,color='red',marker='o',label=label)
    plt.step(range(0,iteration_num-5,1),data50,linewidth=0.35,color='red')


    label='75% Percentile'     
    iteration_num=0
    run=output[index75];
    for X in run:   
        if iteration_num>4:
            dataY75=X[0]  
            # special case condition code fault  
            if iteration_num<40:      
                plt.scatter(iteration_num-5,dataY75, s=s_value_big,marker='^',facecolors='#152CEB')
        iteration_num+=1
    if iteration_num==40:
        plt.scatter(iteration_num-5,dataY75, s=s_value_big,marker='^',facecolors='#152CEB',label=label)
        
    plt.axhline(y = ground_y_max, color = '#C49A17', linestyle = 'dashed',label='Ground Truth Max')

    

    xticks=[0,10,20,30,40,50,55]
    yticks=[-20,-15,-10,-5,0]
    plt.xticks(xticks)
    plt.yticks(yticks)  
    plt.tick_params(direction='in')      
    # plt.xlabel("Number of Iteration", fontdict=label_font)
    # plt.ylabel("Max(μ_D($\mathbf{X}$))", fontdict=label_font)
    plt.xlabel("Iteration")
    plt.ylabel("Max(μ_D($\mathbf{X}$))")
    # plt.ylim(-22,2.5)
    plt.xlim(-1,55)
    
    plt.tight_layout()
    plt.savefig('Learning_curve_Y_scatter_track_X.svg')
    plt.close()

   

    
    sum_CR_X/=99
    sum_CR_Y/=99
    sum_IR_X/=99
    sum_IR_Y/=99
    
    file_name='LOG_TABLE.csv'
    record_data_log_file.log_table(beta,std_noise,sum_IR_X,sum_CR_X,sum_IR_Y,sum_CR_Y,file_name)

    

def plotting(Output,n_lhs_samples,total,bs,start,iterations,ground_y_max,beta,std_noise,filename,ker_variance_filename,ker_lengthscale_filename,model):
    #boxplot

    scatter_25_and_75_percentile_track_X(Output,n_lhs_samples,total,bs,start,iterations,ground_y_max,beta,std_noise,filename,ker_variance_filename)
    
    