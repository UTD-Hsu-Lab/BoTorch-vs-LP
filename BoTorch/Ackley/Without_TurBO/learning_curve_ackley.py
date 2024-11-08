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

import TestFunction
import record_data_log_file
import Norm_Dnorm





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
    x0 = ((x[0]) - 0.20169)**2
    x1 = ((x[1]) - 0.150011)**2
    x2 = ((x[2]) - 0.476874)**2
    x3 = ((x[3]) - 0.275332)**2
    x4 = ((x[4]) - 0.311652)**2
    x5 = ((x[5]) - 0.6573)**2
    #print(np.sqrt(x0+x1+x2+x3+x4+x5))
    return np.sqrt(x0+x1+x2+x3+x4+x5)
   



def euclidean_distance_ackley(x,iteration_num):
  return np.sqrt(np.sum([x[i]**2 for i in range(len(x))]))

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







def scatter_25_and_75_percentile_track_X(Output,n_lhs_samples,total,bs,start,iterations,ground_y_max,beta,std_noise,filename):
    path=os.getcwd() 
    # Value of X for which you want to estimate the mean
    ground_x_max = 0    
    output = []
    # Normalizing Y
    
    ymin=-22.3    
    ymax = 0
    xmax=32.768;xmin=-32.768
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
                i=1
                for x in row[1:6]:
                    row[i]=Norm_Dnorm.denormalizer(row[i],xmax,xmin)   
                    i+=1          
                output[ctr-1].append((row[0],row[1:]))

    total_iter = len(output[0])
  



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
    print('Iterations = ',num_of_iterations)

    #fig features

    fig=plt.figure(figsize=(2.5,2))
    plt.style.use('seaborn-paper')
    s_value=8
    s_value_big=13

    

    


    for iteration_num in range(len(output[0])):
        data = [euclidean_distance_ackley(run[iteration_num][1],iteration_num) for run in output] 
        data=np.array(data)  

        print(data)     
        
        #print("Median value:", median_value)
        #print("Index of the median value:", index,'\t and corresponded data = ',data[index])
        x=iteration_num
        length=len(data)  
        #plt.title(figure_title)   
          
        plt.scatter(np.repeat(x,length), data ,s=s_value,color='#6A6B72',marker='*')
        #plt.scatter(np.repeat(x,length), data ,color='#6A6B72',marker='*') 
        
    
        
       

    # # Taking percentile of 25 and 75 of X and corresponded Y    
    plt.scatter(np.repeat(x,length), data ,s=s_value,color='#6A6B72',marker='*',label='99 LHS for BO models')
    #plt.scatter(np.repeat(x,length), data ,color='#6A6B72',marker='*',label='$\mathbf{X}$ for 99 LHS')
    medians_X=np.median(data)
    indexY=0  
    for d in data:
        if medians_X==d:
            break;
        indexY+=1 
    #calculating y0
    yi=output[indexY][0][0]
    #calculating yf
    yf=output[indexY][num_of_iterations-1][0]


    print('median index =  ',indexY)
   

    
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
        plt.scatter(iteration_num,dataY25,s=s_value_big, marker='v',facecolors='#44F65C')
        #plt.scatter(iteration_num,dataY25,marker='v',facecolors='#44F65C')
    plt.scatter(iteration_num-1,dataY25,s=s_value_big, marker='v',facecolors='#44F65C',label=label)
    #plt.scatter(iteration_num-1,dataY25,marker='v',facecolors='#44F65C',label=label)

    
    index_75th_percentile = int(0.25 * len(data))
    value_75th_percentile = sorted_data[index_75th_percentile]
    index75=0
    for d in data:            
        if value_75th_percentile==d:
            break;
        index75+=1 

    # index50_75=[]
    # index_array=[5,10,15,20,25,30,35,40,45,52,55,60,65,70,58,63,68,73]
    # for i in index_array:
    #     percentage=float(i)/100
    #     temp_index=int((1-percentage)*len(data))
    #     temp_value=sorted_data[temp_index] 
    #     temp_index=0
    #     for d in data:            
    #         if temp_value==d:
    #             break;
    #         temp_index+=1
    #     index50_75.append(temp_index)

    label='50% Percentile'
    data50=[];ACR_X=0
    for iteration_num in range(len(output[0])):  
        dataY=euclidean_distance_ackley(output[indexY][iteration_num][1],iteration_num)   
        data50.append(dataY) 
        plt.scatter(iteration_num,dataY,s=s_value_big,color='red',marker='o')
        #plt.scatter(iteration_num,dataY,color='red',marker='o')
        #plt.step(iteration_num,dataY,color='red')

        #Calculating ACR_X
        
    #ACR_X=ACR_X/num_of_iterations
    #end.......
    plt.scatter(iteration_num,dataY,s=s_value_big,color='red',marker='o',label=label)
    #plt.scatter(iteration_num,dataY,color='red',marker='o',label=label)
    plt.step(range(0,len(output[0]),1),data50,linewidth=0.35,color='red')


    label='75% Percentile'     
    for iteration_num in range(len(output[0])): 
        dataY75=euclidean_distance_ackley(output[index75][iteration_num][1],iteration_num)       
        plt.scatter(iteration_num,dataY75,s=s_value_big, marker='^',facecolors='#152CEB')
        #plt.scatter(iteration_num,dataY75,marker='^',facecolors='#152CEB')
    plt.scatter(iteration_num-1,dataY75,s=s_value_big, marker='^',facecolors='#152CEB',label=label)
    #plt.scatter(iteration_num-1,dataY75,marker='^',facecolors='#152CEB',label=label)

    plt.axhline(y = ground_x_max, color = '#C49A17', linestyle = 'dashed',label='Ground Truth Max')
    label_font = { 'fontsize': 18}
    tick_font = { 'fontsize': 15}
    xticks=[0,10,20,30,40,50]
    yticks=[0,10,20,30,40,50]
    plt.xticks(xticks)
    plt.yticks(yticks)  
    plt.tick_params(direction='in')      
    # plt.xlabel("Number of Iteration", fontdict=label_font)
    # plt.ylabel("Euclidean distance of $\mathbf{X}$", fontdict=label_font)
    plt.xlabel("Iteration")
    plt.ylabel("||$\mathbf{X}^*$ - $\mathbf{X}_{\mathrm{max}}$||")
    # plt.ylim(-0.05,1.5)
    plt.xlim(-1,50)
    plt.ylim(-3.5,50)
    
    legend_font = {'size': 18}
    # plt.legend(prop=legend_font,loc='best')
    # legend = plt.legend()
    # legend.get_frame().set_alpha(0.6)
    # plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.35))
    
    plt.tight_layout()     
    #plt.title('(a)')
    plt.savefig('Learning_curve_X_scatter_track_X.svg')
    plt.close()



    
    

    fig1=plt.figure(figsize=(2.63,2))
    plt.style.use('seaborn-paper')

    num_of_iterations=len(output[0])    
    for iteration_num in range(len(output[0])):
        data = [run[iteration_num][0] for run in output]       
               
        data=np.array(data)             
        x=iteration_num
        length=len(data)       
        plt.scatter(np.repeat(x,length), data ,s=s_value,color='#6A6B72',marker='*') 
        
        
    plt.scatter(np.repeat(x,length), data ,s=s_value,color='#6A6B72',marker='*',label='99 LHS for BO models') 

    
    label='25% Percentile'
    for iteration_num in range(len(output[0])): 
        dataX25=output[index25][iteration_num][0]    
        plt.scatter(iteration_num,dataX25,s=s_value_big, marker='v',facecolors='#44F65C')
    plt.scatter(iteration_num-1,dataX25,s=s_value_big, marker='v',facecolors='#44F65C',label=label)

    label='50% Percentile'
    data50=[];ACR_Y=0
    for iteration_num in range(len(output[0])):  
        dataX=output[indexY][iteration_num][0]
        data50.append(dataX)     
        plt.scatter(iteration_num,dataX,s=s_value_big,color='red',marker='o')
        #plt.step(iteration_num,dataX,color='red')

        #Calculating ACR_Y
        ACR_Y=np.abs(ACR_Y+(ground_y_max-dataX))
    #ACR_Y=ACR_Y/num_of_iterations
    #calculating IR_Y
    IR_Y=ground_y_max-dataX
    #end

    plt.scatter(iteration_num-1,dataX,s=s_value_big,color='red',marker='o',label=label)
    plt.step(range(0,len(output[0]),1),data50,linewidth=0.35,color='red')

    
    
    label='75% Percentile'
    for iteration_num in range(len(output[0])): 
        dataX75=output[index75][iteration_num][0]      
        plt.scatter(iteration_num,dataX75,s=s_value_big, marker='^',facecolors='#152CEB')
    plt.scatter(iteration_num-1,dataX75,s=s_value_big, marker='^',facecolors='#152CEB',label=label)


    
    

    # end..............
        
    #plt.axhline(y = ground_y_max, color = '#C49A17', linestyle = 'dashed',linewidth=3.0,label='Ground Truth Max')
    plt.axhline(y = ground_y_max, color = '#C49A17', linestyle = 'dashed',label='Ground Truth Max')
    label_font = { 'fontsize': 18}
    tick_font = { 'fontsize': 15}
    xticks=[0,10,20,30,40,50]
    yticks=[-20,-15,-10,-5,0]
    plt.xticks(xticks)
    plt.yticks(yticks)  
    plt.tick_params(direction='in')      
    # plt.xlabel("Number of Iteration", fontdict=label_font)
    plt.ylabel("μ_D($\mathbf{X}^*$)")
    plt.xlabel("Iteration")
    #plt.ylabel("Max(y)")
    
    plt.ylim(-22,2.5)
    plt.xlim(-1,50)
    # legend_font = { 'size': 18}
    #plt.legend(prop=legend_font,loc='best')
    
    #plt.title('Learning Curve of Y', fontdict=title_font)
    plt.tight_layout()
    #plt.title('(b)')
    plt.savefig('Learning_curve_Y_scatter_track_X.svg')
    plt.close()


    sum_IR_X=0
    sum_IR_Y=0

    
            

    for run in output:
        data = euclidean_distance_ackley(run[len(output[0])-1][1],iteration_num=49) 
        sum_IR_X+=data
        data = np.abs(ground_y_max-run[len(output[0])-1][0]) 
        #print(data,run[len(output[0])-1][0])
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



def plotting(Output,n_lhs_samples,total,bs,start,iterations,ground_y_max,beta,std_noise,filename):
    
    # # X tracking portion
    index50_75=[]
    index_corresponded_Y,percentile25_X_index,percentile75_X_index=scatter_25_and_75_percentile_track_X(Output,n_lhs_samples,total,bs,start,iterations,ground_y_max,beta,std_noise,filename)
    
    
    