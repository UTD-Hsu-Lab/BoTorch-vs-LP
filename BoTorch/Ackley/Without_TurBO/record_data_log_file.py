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
  
      #f=open(file,'w')
      #f.write('Beta\t')
      #f.write('Std_Noise\t')
      #f.write('SD(Y)\t')       
      #f.write('SD(X)\t')
      #f.write('IR_Y\t')      
      #f.write('ACR_Y\n')
      #f.write('Max_Log_Likelihood_Y\n')


      #f.write('IR_X\t')
      #f.write('ACR_X\t')    
      #f.write('Max_Log_Likelihood_X\n')  
      #f.write('Difference in Median(Y)\t')
      #f.write('Gap(Y)_Median\t')
      
      
      #f.write(str(beta))
      #f.write('\t')
      #f.write(str(std_noise))
      #f.write('\t')
      #f.write(str(np.std([run[iterations - 1][1] for run in output])))
      #f.write('\t')
      #f.write(str(yf - y0))
      #f.write('\t')
      #f.write(str(IR_Y))
      #f.write('\t')
      #f.write(str(ACR_Y))
      #f.write('\t')
      #f.write(str(MLL_Y_R))
      #f.write('\t')

      #f.write(str(IR_X))
      #f.write('\t')
      #f.write(str(ACR_X))
      #f.write('\t')
      #f.write(str(MLL_X_R))
      #f.write('\t')
      #f.write(str(GAP))
      #f.write('\n')
      
      #f.close()

  
  #script_directory = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
  #os.chdir(script_directory)
  

