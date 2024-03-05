import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

def step_function(x):
    return 1 if x >= 0 else 0

def normalize_data(X):
    return X / np.max(X)

def initialize_weights(input_size):
    return np.array([10, 0.2, -0.75])

def train_perceptron(X_train, y_train, alpha=0.05, max_epochs=1000, convergence_threshold=0.002):
    input_size = X_train.shape[1]
    W = initialize_weights(input_size)

    for epoch in range(max_epochs):
        total_error = 0

        for i in range(len(X_train)):
            inputs = np.insert(X_train[i], 0, 1)
            output = step_function(np.dot(W, inputs))
            error = y_train[i] - output
            total_error += error**2
            W += alpha * error * inputs

        if total_error <= convergence_threshold:
            return epoch + 1

    return max_epochs  # If not converged within max_epochs, return max_epochs

if __name__ == "__main__":
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 0, 0, 1])
    X_train_normalized = normalize_data(X_train)

    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    iterations_to_converge = []

    for alpha in learning_rates:
        iterations = train_perceptron(X_train_normalized, y_train, alpha)
        iterations_to_converge.append(iterations)

    # Plotting the number of iterations against learning rates
    plt.plot(learning_rates, iterations_to_converge, marker='o')
    plt.xlabel('Learning Rate')
    plt.ylabel('Iterations to Converge')
    plt.title('Iterations to Converge vs Learning Rate')
    plt.grid(True)
    plt.show()
