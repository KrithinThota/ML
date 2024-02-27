import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
num_points = 20
X = np.random.randint(1, 11, size=num_points)
Y = np.random.randint(1, 11, size=num_points)

classes = np.where(X + Y > 15, 1, 0)

class0_X, class0_Y = X[classes == 0], Y[classes == 0]
class1_X, class1_Y = X[classes == 1], Y[classes == 1]

plt.figure(figsize=(8, 6))
plt.scatter(class0_X, class0_Y, color='blue', label='Class 0')
plt.scatter(class1_X, class1_Y, color='red', label='Class 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Training Data')
plt.legend()
plt.grid(True)
plt.show()
