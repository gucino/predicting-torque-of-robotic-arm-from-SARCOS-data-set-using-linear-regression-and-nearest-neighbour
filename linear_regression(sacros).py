# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 20:14:25 2020

@author: Tisana
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 13:40:22 2020

@author: Tisana
"""


#sacros problem
import csv
import numpy as np
import matplotlib.pyplot as plt
import timeit
start = timeit.default_timer()

#input
with open('sarcos_inv.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    data_set=[]
    for each_row in csv_reader:
        data_set.append([float(i) for i in each_row])
        
#convert to x and y
data_set_array=np.array(data_set)
x_raw=data_set_array[:,:21]
y_raw=data_set_array[:,21:]


total_count = x_raw.shape[0]


# Shuffle the data to avoid any ordering bias..
np.random.seed(0)
shuffle = np.random.permutation(total_count)

x = x_raw[shuffle]
y = y_raw[shuffle]

def normalise_data(x_unnormalised):
    b=np.mean(x_unnormalised)
    a=np.std(x_unnormalised)
    x_normalised=(x_unnormalised-b)/a
    return x_normalised
############################################################################
############################################################################  
############################################################################  
#for each regression model
    
def MSE(actual_y,predicted_y): #both must have same dimension
    SSE=sum((actual_y-predicted_y)**2)
    num_observation=(predicted_y.shape)[0]
    if num_observation!=0:
        
        
        MSE=SSE/num_observation
        
    else:
        MSE=0
    return MSE

#covariance function
def covariance(x,y):
    x_bar=np.mean(x)
    y_bar=np.mean(y)
    n=len(x)
    x_minus_x_bar=x-x_bar
    y_minus_y_bar=y-y_bar
    a=x_minus_x_bar*y_minus_y_bar
    cov=sum(a)/(n-1)
    return cov

#function that find slope and intercept
def slope_and_interept(x,y): #as array (y must be one dimension)
    left=[]                  #x must be 2 dimension
    right=[]
    x_bar_list=[]
    for i in range((x.shape)[1]):
        #print("feature : ",i)
        this_x=x[:,i]
        right_cov=covariance(y,this_x)
        right.append(right_cov)
        local_left_list=[]
        for j in range((x.shape)[1]):
            left_cov=covariance(x[:,j],this_x)
            local_left_list.append(left_cov)
        left.append(local_left_list)
        
        x_bar=np.mean(x[:,i])
        x_bar_list.append(x_bar)
    x_bar_list=np.array(x_bar_list)
    #solve for slope
    a=np.array(left)
    b=np.array(right)
    w_list=np.linalg.solve(a,b) 
    
    #solve for intercept
    y_bar=np.mean(y)
    c=y_bar-sum(w_list*x_bar_list)
    
    return w_list,c



############################################################################  
############################################################################  
############################################################################      
x=normalise_data(x)
y=normalise_data(y)

#split into 6 folds
test_split_index_list=[[0,7414],
                       [7414,14828],
                       [14828,22242],
                       [22242,29656],
                       [29656,37070],
                       [37070,44484]]

rmse_list=[]
mae_list=[]
mse_list=[]
for i in range(0,len(test_split_index_list)):
    print("fold : ",i)

    x_train=np.array(x[0:test_split_index_list[i][0]].tolist()+x[test_split_index_list[i][1]:len(x)].tolist())
    x_test=x[test_split_index_list[i][0]:test_split_index_list[i][1]]
    
    y_train=np.array(y[0:test_split_index_list[i][0]].tolist()+y[test_split_index_list[i][1]:len(y)].tolist())
    y_test=y[test_split_index_list[i][0]:test_split_index_list[i][1]]
    
    #find w(slope) and c (intercept)
    y_train=np.squeeze(y_train)
    w_list,c=slope_and_interept(x_train,y_train)
    
    #prediction
    y_pred=(w_list*x_test).sum(axis=1)+c
    y_pred=y_pred[:,np.newaxis]
    
    #error
    mse=MSE(y_test,y_pred)
    mse_list.append(mse)
    rmse=np.sqrt(mse)
    rmse_list.append(rmse) 
    abs_diff=abs(y_test-y_pred)
    mae=abs_diff/((y_pred.shape)[0])
    mae_list.append(mae)

############################################################################  
############################################################################  
############################################################################   
    
  
#calculate mean ans sd
mean_mse=np.mean(mse_list)
sd_mse=np.std(mse_list)
mean_rmse=np.mean(rmse_list)
sd_rmse=np.std(rmse_list)
mean_mae=np.mean(mae_list)
sd_mae=np.std(mae_list)
print("mean mse : ",mean_mse)
print("sd mse : ",sd_mse)
print("")
print("mean rmse : ",mean_rmse)
print("sd rmse : ",sd_rmse)
print("")
print("mean mae : ",mean_mae)
print("sd mae : ",sd_mae)
print("")
############################################################################  
############################################################################  
############################################################################    

stop = timeit.default_timer()
solving_time=stop-start
print("solving time : ",solving_time)