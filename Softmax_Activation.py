# Activation function like sigmoid, tanh,ReLU, Leaky ReLU and parametric ReLU , ELU, softmax
# for hidden layers , we use Relu or its varints 
# for output layer, we use sigmoid when binray-class classification problem
# for output layer, we use softmax when multi-class classification problem

# softmax activation function
# process
# input -> exponentiate -> normalize -> output
# softmax (expronentiate + normalize)

# step-3 now understand for batch of inputs
import math
import numpy as np

layer_outputs = [[4.8,1.21,2.385],
                 [8.9,-1.81,0.2],
                 [1.41,1.051,0.026]]

exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)

# step-2 softmax using numpy for single inputs
'''
import math
import numpy as np
layer_outputs = [4.8,1.21,2.385]

exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values)
print(norm_values)
print(sum(norm_values))
'''

# step-1 understand softmax in python code
'''
E = math.e
# E = 2.71828182846

exp_values = []
layer_outputs = [4.8,1.21,2.385]

for output in layer_outputs:
    exp_values.append(E**output)

print(exp_values)

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)

print(norm_values)
print(sum(norm_values))
'''