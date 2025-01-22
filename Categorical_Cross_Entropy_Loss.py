#  loss functions 
#  for regression -  MSE, MAE, RMSE and huber loss
#  for classification - binary cross entropy and categorical cross entropy

# categorical cross entropy for multi-class classifiction problem
#  - sum( log(predictions) * real_target)

import math

# in output layer applied softmax activation then get values
softmax_output = [0.7,0.1,0.2] #predictions values
target_output = [1,0,0] #real target values

loss = -(
    math.log(softmax_output[0])*target_output[0]+
    math.log(softmax_output[1])*target_output[1]+
    math.log(softmax_output[2])*target_output[2]
)

print(loss)