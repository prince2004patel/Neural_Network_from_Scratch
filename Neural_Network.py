import numpy as np
from Generated_Data import spiral_data 

# Set seed for reproducibility
np.random.seed(0)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.input = inputs  # Store the input for the backward pass
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.input.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinput = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.input = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinput = dvalues.copy()
        self.dinput[self.input <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # Overflow prevention
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        self.dinput = dvalues.copy()

        if len(y_true.shape) == 2:
            self.dinput = self.dinput - y_true
        else:
            for i in range(samples):
                self.dinput[i, y_true[i]] -= 1

        self.dinput = self.dinput / samples

class Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate  # Keep rate

    def forward(self, inputs):
        self.input = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinput = dvalues * self.binary_mask

class Loss:
    def calculate(self, output, y):
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

    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 2:
            self.dinput = y_pred_clipped - y_true
        else:
            self.dinput = y_pred_clipped
            self.dinput[range(samples), y_true] -= 1

        self.dinput = self.dinput / samples

class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layer):
        self.t += 1

        if layer not in self.m:
            self.m[layer] = np.zeros_like(layer.dweights)
            self.v[layer] = np.zeros_like(layer.dweights)

        self.m[layer] = self.beta1 * self.m[layer] + (1 - self.beta1) * layer.dweights
        self.v[layer] = self.beta2 * self.v[layer] + (1 - self.beta2) * (layer.dweights ** 2)

        m_hat = self.m[layer] / (1 - self.beta1 ** self.t)
        v_hat = self.v[layer] / (1 - self.beta2 ** self.t)

        layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        layer.biases -= self.learning_rate * layer.dbiases
'''
# Training Data
X, y = spiral_data(points=100, classes=3) 

# Create layers
dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dropout1 = Dropout(rate=0.1)  # Dropout after first layer

dense2 = Layer_Dense(64, 3)
activation2 = Activation_Softmax()

# Loss function
loss_function = Loss_CategoricalCrossEntropy()

# Optimizer
optimizer = Optimizer_Adam(learning_rate=0.001)

# Training Loop
for epoch in range(5001):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)  # Apply dropout

    dense2.forward(dropout1.output)
    activation2.forward(dense2.output)

    # Calculate loss
    loss = loss_function.calculate(activation2.output, y)

    # Backward pass
    activation2.backward(activation2.output, y)
    dense2.backward(activation2.dinput)

    dropout1.backward(dense2.dinput)  # Backpropagate through dropout
    activation1.backward(dropout1.dinput)
    dense1.backward(activation1.dinput)

    # Update weights
    optimizer.update(dense1)
    optimizer.update(dense2)

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# After training, print predictions
dense1.forward(X)
activation1.forward(dense1.output)
dropout1.forward(activation1.output)
dense2.forward(dropout1.output)
activation2.forward(dense2.output)

# Get predicted class labels by taking the index of the maximum probability
predictions = np.argmax(activation2.output, axis=1)

# Compare predictions with true labels (y)
accuracy = np.mean(predictions == y)
print(f"Final Accuracy: {accuracy * 100:.2f}%")
'''
