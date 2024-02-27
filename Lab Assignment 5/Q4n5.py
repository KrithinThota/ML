import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)
X_train = np.random.uniform(0, 10, size=(100, 2))
y_train = np.random.randint(0, 2, size=(100,))

x_values = np.arange(0, 10.1, 0.1)
y_values = np.arange(0, 10.1, 0.1)
X_test = np.array([[x, y] for x in x_values for y in y_values])

def knn(X_train, y_train, X_test, k=3):
    X_train = tf.constant(X_train, dtype=tf.float32)
    X_test = tf.constant(X_test, dtype=tf.float32)
    distances = tf.reduce_sum(tf.square(tf.subtract(X_train, tf.expand_dims(X_test, axis=1))), axis=2)
    nearest_indices = tf.argsort(distances, axis=1)[:, :k]
    nearest_labels = tf.gather(y_train, nearest_indices)
    predicted_labels = tf.squeeze(tf.reduce_max(nearest_labels, axis=1))
    return predicted_labels.numpy()

predicted_classes = knn(X_train, y_train, X_test)

plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=predicted_classes, cmap='coolwarm', s=10)
plt.title('kNN Classification of Test Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Predicted Class')
plt.show()
