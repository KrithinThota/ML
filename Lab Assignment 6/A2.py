import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

def bipolar_step_function(x):
    return 1 if x >= 0 else -1

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def relu_function(x):
    return max(0, x)

def normalize_data(X):
    return X / np.max(X)

def initialize_weights(input_size):
    return np.random.randn(input_size + 1)

def train_perceptron(X_train, y_train, activation_function, alpha=0.05, max_epochs=1000, convergence_threshold=0.002, validation_set=None):
    input_size = X_train.shape[1]
    W = initialize_weights(input_size)
    errors = []

    for epoch in range(max_epochs):
        total_error = 0

        for i in range(len(X_train)):
            inputs = np.insert(X_train[i], 0, 1)
            output = activation_function(np.dot(W, inputs))
            error = y_train[i] - output
            total_error += error**2
            W += alpha * error * inputs

        errors.append(total_error)

        if total_error <= convergence_threshold:
            logging.info(f"Converged at epoch {epoch + 1}.")
            break

        if validation_set:
            X_val, y_val = validation_set
            val_errors = [np.sum((y_val - activation_function(np.dot(W, np.insert(x, 0, 1)))) ** 2) for x in X_val]
            val_error = np.mean(val_errors)
            logging.info(f"Epoch {epoch + 1}, Validation Error: {val_error}")

    return W, errors

def plot_errors(errors):
    plt.plot(range(1, len(errors) + 1), errors, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Epochs vs Error')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 0, 0, 1])
    X_train_normalized = normalize_data(X_train)

    activation_functions = {
        'Bi-Polar Step': bipolar_step_function,
        'Sigmoid': sigmoid_function,
        'ReLU': relu_function
    }

    for name, activation_function in activation_functions.items():
        logging.info(f"Training with {name} activation function.")
        W_final, errors = train_perceptron(X_train_normalized, y_train, activation_function)
        plot_errors(errors)
