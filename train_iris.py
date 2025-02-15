import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from Neural_Network import Layer_Dense, Activation_ReLU, Activation_Softmax, Loss_CategoricalCrossEntropy, Optimizer_Adam, Dropout

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# One-hot encode the target labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize layers
dense1 = Layer_Dense(4, 5)  # 4 features (from the Iris dataset), 5 neurons in the first layer
activation1 = Activation_ReLU()
dropout1 = Dropout(rate=0.1)  # Dropout for the first hidden layer

dense2 = Layer_Dense(5, 3)  # 5 neurons in the previous layer, 3 output classes (Iris species)
activation2 = Activation_Softmax()

# Loss function
loss_function = Loss_CategoricalCrossEntropy()

# Optimizer
optimizer = Optimizer_Adam(learning_rate=0.01)

# Training Loop
for epoch in range(10001):
    # Forward pass
    dense1.forward(X_train)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)  # Apply dropout to the first hidden layer's activations
    
    dense2.forward(dropout1.output)
    activation2.forward(dense2.output)
    # No dropout after softmax (output layer)

    # Calculate loss
    loss = loss_function.calculate(activation2.output, y_train)
    
    # Backward pass
    activation2.backward(activation2.output, y_train)
    dense2.backward(activation2.dinput)
    
    dropout1.backward(dense2.dinput)  # Backprop through dropout1
    activation1.backward(dropout1.dinput)
    dense1.backward(activation1.dinput)
    
    # Update weights with Adam optimizer
    optimizer.update(dense1)
    optimizer.update(dense2)
    
    # Print progress every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# After training, calculate accuracy
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Get predicted class labels by taking the index of the maximum probability
predictions = np.argmax(activation2.output, axis=1)

# Convert one-hot encoded true labels to class labels
y_test_labels = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = np.mean(predictions == y_test_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")
