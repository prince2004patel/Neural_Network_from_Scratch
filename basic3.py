# understanding layers, batch and object

# step 1 , understand this frist 

# Problem and Solution:
# Previously in basic2.py, the dot product between weights and inputs worked correctly. 
# The weights had a shape of (3, 4), so the inputs needed to have a shape of (4,) or (4, 1). 
# However, when passing a batch of inputs, the input shape changed, causing an error.
# To fix this, we transpose the inputs. For a batch, inputs might have a shape of (3, 4), 
# and transposing them gives a shape of (4, 3). 
# Now, the dot product works: weights with shape (3, 4) and inputs with shape (4, 3) result in an output of shape (3, 3).

'''
# now it have 4 neurons in input layers and hidden layer have 3 and second layer(output) have 3 neurons
import numpy as np

inputs = [
    [1,2,3,2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]
]
# it is matrix as (3,4) shape
# we pass 3 sample who have 4 neurons input as batch

weights =[
    [0.2,0.8,-0.5,1.0],
    [0.5,-0.91,0.26,-0.5],
    [-0.26,-0.27,0.17,0.87]
]
# it is matrix as (3,4) shape

biases= [2, 3, 0.5]

weights2 =[
    [0.1,-0.14,0.5],
    [-0.5,0.12,-0.33],
    [-0.44,0.73,-0.13]
]
# it is matrix as (3,4) shape

biases2 = [-1, 2, -0.5]

layer1_output = np.dot(inputs,np.array(weights).T) + biases

layer2_output = np.dot(layer1_output,np.array(weights2).T) + biases2

print(layer2_output)
'''

# step2,

import numpy as np

np.random.seed(0)

X = [
    [1,2,3,2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]
]

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        # make neurons small as small values like between 0 to 1 or -1 to 1
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)
# layer1 input is X as 4 and layer1 output is 5 in hidden layers
# layer2 input is layer1's output as 5 and layer2 output is 2
# so it make 4,5,2 of input,hidden,output layers

layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)