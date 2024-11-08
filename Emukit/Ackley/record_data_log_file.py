import subprocess
import numpy as np
import time
import math
from matplotlib import pyplot as plt
import GPy
import csv
import os


# CSV output
def record_csv_X(output, filename):
  with open(filename, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile, delimiter=',')
      run = 1
      for i in output:
        writer.writerow([run])
        for tple in i:
          writer.writerow((tple))
        run += 1
def record_csv(output, filename):
  with open(filename, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile, delimiter=',')
      run = 1
      for i in output:
        writer.writerow([run])
        for tple in i:
          writer.writerow(np.append(tple[1], tple[0]))
        run += 1

def record_csv_variance(output, filename):
  with open(filename, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile, delimiter=',')
      run = 1
      for i in output:
        writer.writerow([run])
        for tple in i:          
          writer.writerow(np.append(tple[0],tple[1]))
        run += 1

def record_csv_lengthscale(output, filename):
  with open(filename, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile, delimiter=',')
      run = 1      
      for i in output:
        writer.writerow([run])        
        for tple in i:                  
          writer.writerow(tple)
        run += 1
def record_csv_model(output, filename):
    with open(filename, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile, delimiter=',')
      run = 1     
         
      for i in output:
        writer.writerow(np.append([run],[run]))         
        limit=np.size(i)           
        for j in range(limit):                                
          writer.writerow(i[0][j])
        run += 1
  
        

def record_variance(output, filename = 'bo.csv'):
  f=open(filename,'w')
  #writer = csv.writer(csvfile, delimiter=',')
  run = 1
  for i in output:
      f.write(str(run))
      f.write('\n')      
      for tple in i:
        f.write(str(tple))
        f.write('\n')
      run += 1

def record_LHS(output, filename,run):
  if os.path.exists(filename):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([run])
        for i in output:        
           writer.writerow((i[0], i[1],i[2],i[3],i[4],i[5]))
  else:
     with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([run])
        for i in output:             
          writer.writerow((i[0], i[1],i[2],i[3],i[4],i[5]))
        


def log_table(beta,std_noise,IR_Y,ACR_Y,IR_X,ACR_X,file_name):
  
  file=file_name
  #os.chdir()  
  with open(file,'w',newline='') as csvfile:
      writer = csv.writer(csvfile, delimiter=',')
      #writer.writerow(('Beta','Std_Noise','IR_Y','ACR_Y','Max_Log_Likelihood_Y','IR_X','ACR_X','Max_Log_Likelihood_X','Gap'))
      writer.writerow((beta,std_noise,IR_Y,ACR_Y,IR_X,ACR_X))

def record_parity_plot_data(X,Y_truth,Y_pred,filename,run):
  if os.path.exists(filename):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([run])
        for x,y_truth,y_pred in zip(X,Y_truth,Y_pred):        
          writer.writerow((x[0], x[1],x[2],x[3],x[4],x[5],y_truth,y_pred))
  else:
     with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([run])
        for x,y_truth,y_pred in zip(X,Y_truth,Y_pred):             
          writer.writerow((x[0], x[1],x[2],x[3],x[4],x[5],y_truth,y_pred))

  

