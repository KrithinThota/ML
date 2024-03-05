import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        # Initialize weights and bias
        self.weights = np.random.randn(X.shape[1])
        self.bias = np.random.randn(1)

        for epoch in range(self.epochs):
            # Forward pass
            output = self.sigmoid(np.dot(X, self.weights) + self.bias)
            
            # Compute error
            error = y - output
            
            # Update weights and bias
            self.weights += self.learning_rate * np.dot(X.T, error)
            self.bias += self.learning_rate * np.sum(error)

    def predict(self, X):
        output = self.sigmoid(np.dot(X, self.weights) + self.bias)
        return np.where(output >= 0.5, 1, 0)

# Load customer data
customer_data = {
    'Candies': [10, 5, 8, 12, 3, 15, 7, 6, 9, 4],
    'Mangoes_Kg': [2, 1, 1.5, 3, 0.5, 4, 1.8, 2.2, 3.5, 0.8],
    'Milk_Packets': [2, 1, 1, 3, 1, 4, 2, 3, 3, 1],
    'Payment_Rs': [100, 50, 80, 120, 30, 150, 70, 60, 90, 40],
    'High_Value_Tx': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No']
}

# Convert to DataFrame
df = pd.DataFrame(customer_data)

# Encode target variable
df['High_Value_Tx'] = df['High_Value_Tx'].map({'Yes': 1, 'No': 0})

# Split features and target
X = df.drop('High_Value_Tx', axis=1)
y = df['High_Value_Tx']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the perceptron model
perceptron = Perceptron(learning_rate=0.1, epochs=1000)
perceptron.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = perceptron.predict(X_train_scaled)
y_pred_test = perceptron.predict(X_test_scaled)

# Evaluate model performance
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
