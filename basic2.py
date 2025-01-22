# using numpy's dot prodcut calculate the z (input * weight + bias) 

import numpy as np

inputs = [1,2,3,2.5]
# it is vector as (4,) shape

weights =[
    [0.2,0.8,-0.5,1.0],
    [0.5,-0.91,0.26,-0.5],
    [-0.26,-0.27,0.17,0.87]
]
# it is matrix as (3,4) shape

biases= [2, 3, 0.5]

# exmaple, 2,4 then second start with 4,2 then output is 2,2 for dot product
# it must be weights first  right now
# weights have 3,4 and inputs  have 4,1 so we get 3,1

output = np.dot(weights,inputs) + biases
print(output)
# 4 in input layer and 3 in output layer

'''
layer_outputs = []
for neuron_weights , neuron_bias in zip(weights,biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)
'''