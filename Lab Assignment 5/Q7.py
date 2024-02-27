import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(0)
X_train = np.random.uniform(0, 10, size=(100, 2))
y_train = np.random.randint(0, 2, size=(100,))

param_grid = {'n_neighbors': list(range(1, 21))}

knn = KNeighborsClassifier()

random_search = RandomizedSearchCV(knn, param_distributions=param_grid, n_iter=10, cv=5, random_state=0)
random_search.fit(X_train, y_train)

best_k = random_search.best_params_['n_neighbors']

print("Best value of k found:", best_k)
