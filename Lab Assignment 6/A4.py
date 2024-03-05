import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

def step_function(x):
    return 1 if x >= 0 else 0

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

def train_perceptron(X_train, y_train, activation_function=step_function, alpha=0.05, max_epochs=1000, convergence_threshold=0.002):
    input_size = X_train.shape[1]
    W = initialize_weights(input_size)

    for epoch in range(max_epochs):
        total_error = 0

        for i in range(len(X_train)):
            inputs = np.insert(X_train[i], 0, 1)
            output = activation_function(np.dot(W, inputs))
            error = y_train[i] - output
            total_error += error**2
            W += alpha * error * inputs

        if total_error <= convergence_threshold:
            return epoch + 1

    return max_epochs  # If not converged within max_epochs, return max_epochs

def plot_iterations_vs_learning_rate(X_train, y_train):
    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    iterations_to_converge = []

    for alpha in learning_rates:
        iterations = train_perceptron(X_train, y_train, alpha=alpha)
        iterations_to_converge.append(iterations)

    # Plotting the number of iterations against learning rates
    plt.plot(learning_rates, iterations_to_converge, marker='o')
    plt.xlabel('Learning Rate')
    plt.ylabel('Iterations to Converge')
    plt.title('Iterations to Converge vs Learning Rate')
    plt.grid(True)
    plt.show()

def compare_activation_functions(X_train, y_train):
    activation_functions = {
        'Step': step_function,
        'Bi-Polar Step': bipolar_step_function,
        'Sigmoid': sigmoid_function,
        'ReLU': relu_function
    }

    for name, activation_function in activation_functions.items():
        logging.info(f"Training with {name} activation function.")
        iterations = train_perceptron(X_train, y_train, activation_function)
        print(f"Iterations to converge with {name} activation function: {iterations}")

if __name__ == "__main__":
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train_xor = np.array([0, 1, 1, 0])  # Target labels for XOR gate
    X_train_normalized = normalize_data(X_train)

    # Exercise A1: Training a Perceptron for XOR Gate Logic with Step Activation Function
    iterations_step = train_perceptron(X_train_normalized, y_train_xor)
    print(f"Iterations to converge with Step activation function: {iterations_step}")

    # Exercise A2: Varying Learning Rates for XOR Gate Logic
    plot_iterations_vs_learning_rate(X_train_normalized, y_train_xor)

    # Exercise A3: Activation Function Comparison for XOR Gate Logic
    compare_activation_functions(X_train_normalized, y_train_xor)
