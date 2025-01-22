# understand Optimizer
# optimizer like gradient decent, SGD,  mini batch SGD, SGD with momentum , Adagrad, Adadelta and RMSPROP
# but we use Adam

import numpy as np

# Adam optimizer parameters
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
learning_rate = 0.01

# Sample layer outputs (batch of inputs)
layer_outputs = np.array([[4.8, 1.21, 2.385],
                          [8.9, -1.81, 0.2],
                          [1.41, 1.051, 0.026]])

# Number of neurons
num_neurons = layer_outputs.shape[1]

# Initial weights, m, and v for each neuron
initial_weights = np.full(num_neurons, 0.5)
m = np.zeros(num_neurons)
v = np.zeros(num_neurons)

# Sample gradients for each weight of each neuron (normally calculated during backpropagation)
gradients = np.array([0.1, -0.2, 0.05])  # Replace with actual gradients from backpropagation

# Time step t = 1
t = 1

# Step 1: Calculate first and second moment estimates for each neuron
m_t = beta1 * m + (1 - beta1) * gradients
v_t = beta2 * v + (1 - beta2) * (gradients ** 2)

# Step 2: Bias correction
m_hat = m_t / (1 - beta1 ** t)
v_hat = v_t / (1 - beta2 ** t)

# Step 3: Update weights for each neuron
updated_weights = initial_weights - (learning_rate * m_hat) / (np.sqrt(v_hat) + epsilon)

# Print the results
print(f"Initial weights: {initial_weights}")
print(f"Gradients: {gradients}")
print(f"First moment (m_t): {m_t}")
print(f"Second moment (v_t): {v_t}")
print(f"Bias-corrected first moment (m_hat): {m_hat}")
print(f"Bias-corrected second moment (v_hat): {v_hat}")
print(f"Updated weights: {updated_weights}")

'''
Parameters :-

beta1 and beta2: These are the decay rates for the first and second moment estimates.
epsilon: A small value to prevent division by zero.
learning_rate: The step size for updating the weights.
'''

'''
Initial Values :-

initial_weight: The starting weight.
initial_m and initial_v: Initial values for the first and second moment estimates, usually set to zero.
gradient: The gradient of the loss function with respect to the weights.
'''

'''
Steps:-

Calculate the first moment (m_t) and second moment (v_t).
Apply bias correction to get m_hat and v_hat.
Update the weight using the Adam update rule.
'''