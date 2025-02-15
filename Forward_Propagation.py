# understanding activation functions and final forward propagation will complete

import numpy as np
from Generated_Data import spiral_data

np.random.seed(0)

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # overflow pervention
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) 
        self.output = probabilities

class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_likelihoods = -np.log(correct_confidences)
        return negative_likelihoods

X,y = spiral_data(points=100,classes=3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss",loss)

# Finally, in the forward propagation step:
# 1 -  We calculate z (inputs * weights + bias) for each neuron.
# 2 - Apply the activation function (ReLU or its variants) for the hidden layer.
# 3.1 - Use the softmax function for the output layer to handle multi-class classification.
# 3.2 - if binary-class classification problem then use sigmoid activation function for output layer
# 4.1 - identify problem is classification or regression if regression use MSE, MAE, RMSE, huber loss
# 4.2 - if binary-class classification problem then use binary cross entropy
# 4.3 - Compute the loss using categorical cross-entropy, suitable for classification problems.


# ---------------------------------------------------------------------
# example of ReLU in python
# inputs = [0,2,-1,3,3,-2.7,1.1,2.2,-100]
# output = []

# for i in inputs:
#     if i > 0:
#         output.append(i)
#     elif i <= 0:
#         output.append(0)

# print(output)

