import numpy as np
from collections import Counter

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def k_nearest_neighbors(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = [euclidean_distance(train_point, test_point) for train_point in X_train]
        nearest_neighbors_indices = np.argsort(distances)[:k]
        nearest_neighbors_labels = [y_train[i] for i in nearest_neighbors_indices]
        most_common_label = Counter(nearest_neighbors_labels).most_common(1)[0][0]
        predictions.append(most_common_label)
    return predictions

X_train = np.array([[1, 2], [2, 3],[3,2], [3, 4], [4, 5], [5, 6]])
y_train = np.array([0, 0, 1, 1, 1])
X_test = np.array([[2, 2], [3, 3]])

k = 3
predictions = k_nearest_neighbors(X_train, y_train, X_test, k)
print("Predictions:", predictions)
