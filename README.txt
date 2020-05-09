*******************************************************************
This code was developed for ML1 Lab 5  project
as part of CM50264 Machine learning 1 module university of Bath
*******************************************************************

#purpose
This code aim to predict the torque of one motor of a robotic arm 
from physical joint detail using 2 regression models:
1.linear regression
2.nearest neighbour

#data set
The data set used in this model is SARCOS data set.
which contain 44482 observations and 22 columns. 
The task is to predict the 22th column (torque) from first 21 columns (physical joint detail) 
http://gaussianprocess.org/gpml/data/

#######################################################################
1.linear regression

#result after performing 6 fold cross validation
Mean MSE :0.07369028
SD MSE : 0.002160
Mean RMSE : 0.2714305043372332 
SD RMSE :0.003970436759044734
Mean MAE : 2.6023116958256493e05 
SD MAE :2.5756852105567305e05 
Solving time (second) :20.26 

#######################################################################

#######################################################################
2.nearest neighbour

#hyper-parameter
The number of neighbors is selected according to mean squared error.
three neighbors is selected  as shown in "hyperparameter_selection.png"
 
#result after performing 6 fold cross validation
Mean MSE :0.0618180
SD MSE :  0.001136
Mean RMSE : 0.2486218334534131
SD RMSE :0.002279560864940570
Mean MAE :2.2745908059865487e05 
SD MAE : 2.4642528493716392e05 
Solving time (second) : 395.60801549999996

#######################################################################








