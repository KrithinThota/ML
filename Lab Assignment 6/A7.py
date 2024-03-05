import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.05):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_input_hidden = np.random.randn(self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden_output = np.random.randn(self.output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = self.sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output)
        return self.output
    
    def backward(self, X, y):
        error = y - self.output
        delta_output = error * self.sigmoid_derivative(self.output)
        
        error_hidden = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases        self.weights_hidden_output += self.hidden_output.T.dot(delta_output) * self.learning_rate
        self.bias_hidden_output += np.sum(delta_output) * self.learning_rate
        self.weights_input_hidden += X.T.dot(delta_hidden) * self.learning_rate
        self.bias_input_hidden += np.sum(delta_hidden) * self.learning_rate
        
    def train(self, X, y, epochs=1000, convergence_threshold=0.002):
        for epoch in range(epochs):
            output = self.forward(X)
            
            self.backward(X, y)
            
            error = np.mean(np.square(y - output))
            
            if error <= convergence_threshold:
                print(f"Converged at epoch {epoch + 1}.")
                break
                
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}, Error: {error}")
                
        else:
            print("Maximum number of epochs reached. Did not converge.")
        
    def predict(self, X):
        return np.round(self.forward(X))

import numpy as np

# Generate random binary features
np.random.seed(42)
X_train = np.random.randint(0, 2, size=(1000, 2))

# Compute labels for AND gate logic
y_train = np.array([np.prod(x) for x in X_train]).reshape(-1, 1)

nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.05)
nn.train(X_train, y_train)

print("Predictions:")
print(nn.predict(X_train))
