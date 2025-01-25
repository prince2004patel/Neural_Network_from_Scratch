'''
Dropout is a regularization technique used in neural networks to prevent overfitting.
It works by randomly "dropping out" (setting to zero) a fraction of the neurons during the forward pass of training.
This prevents the network from becoming too reliant on specific neurons and forces it to learn more robust features
'''

import numpy as np

class Dropout:
    def __init__(self, rate):
        # Dropout rate: fraction of neurons to drop
        self.rate = rate

    def forward(self, inputs, training=True):
        if not training:
            self.output = inputs  # During testing, don't apply dropout
            return

        # Generate a dropout mask: 1s and 0s based on the rate
        self.mask = np.random.rand(*inputs.shape) > self.rate
        
        # Apply the mask to the inputs and scale them
        self.output = inputs * self.mask / (1 - self.rate)

    def backward(self, dvalues):
        # Backpropagate through the dropout mask
        self.dinput = dvalues * self.mask


# Example usage
np.random.seed(0)

# Simulate input data
inputs = np.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]])

# Initialize Dropout layer with a rate of 0.5 (50% neurons dropped)
dropout_layer = Dropout(rate=0.5)

# Forward pass during training
dropout_layer.forward(inputs, training=True)
print("Output after applying dropout during training:")
print(dropout_layer.output)

# Forward pass during testing (no dropout applied)
dropout_layer.forward(inputs, training=False)
print("\nOutput during testing (no dropout):")
print(dropout_layer.output)
