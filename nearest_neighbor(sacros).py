# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 20:54:20 2020

@author: Tisana
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:00:52 2020

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
size_of_train_set=0.8
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

#function that calculate euclidean distance for this examplar to all examplars
def euclidean(point1,set_of_point2): #as array
    minus=point1-set_of_point2
    square=minus**2
    sum_of_square=square.sum(axis=1)
    euclidean_distance=np.sqrt(sum_of_square)
    return euclidean_distance

#nearest neighbor function
def y_pred_nearest_neighbor(x_train,y_train,x_test,k):
    y_pred_list=[]
    for i in range(0,(x_test.shape)[0]):
        #print(i)
        point1=x_test[i,:]
        
        #find euclidean distance for all observation from train set
        set_of_point2=x_train
        distance_list=euclidean(point1,set_of_point2)
        index_min_to_max_list=np.argsort(distance_list)
        
        #only k nearest neighbor
        index_min_to_max_list=index_min_to_max_list[:k].tolist()
        #prediction
        y_pred=sum(y_train[index_min_to_max_list])/(len(index_min_to_max_list))
        y_pred=y_pred[0]
        y_pred_list.append(y_pred)
    y_pred_list=np.array(y_pred_list)
    y_pred_list=y_pred_list[:,np.newaxis]
    return y_pred_list

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
    
    #split validation set
    train_size=x_train.shape[0]
    validation_split=int(train_size * size_of_train_set)
    
    x_train_after_validate=x_train[:validation_split]
    y_train_after_validate=y_train[:validation_split]
    
    x_validate=x_train[validation_split:]
    y_validate=y_train[validation_split:]
 

    
    #find optimal number of neighbor
    '''
    mse_list_local=[]
    k_list=[]
    for k in range(1,50,2):
        print(k, "neighbour(s)")
        y_pred=y_pred_nearest_neighbor(x_train_after_validate,y_train_after_validate,x_validate,k)
    
        mse=MSE(y_validate,y_pred)

        print("  mse : ",mse)

        mse_list_local.append(mse)
        k_list.append(k)
      
    plt.figure("find optimal number of neighbor")  
    plt.plot(k_list,mse_list_local,label="mse")  
    plt.legend() 
    plt.savefig("mse")
     
    plt.show()
    k=int(input("num neighbor : "))
    '''
    #prediction
    k=3
    y_pred=y_pred_nearest_neighbor(x_train_after_validate,y_train_after_validate,x_test,k)
    
    
    
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