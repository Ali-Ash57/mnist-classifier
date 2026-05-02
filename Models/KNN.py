import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float32)
        self.y_train = np.asarray(y, dtype=np.int64)


        # precompute norms for fast distance calculation
        self.X_train_norm = np.sum(self.X_train ** 2, axis=1)

    def _euclidean_distances(self, x):
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2a·b
        x_norm = np.sum(x ** 2)
        return self.X_train_norm + x_norm - 2 * np.dot(self.X_train, x)

    def _predict_single(self, x):
        distances = self._euclidean_distances(x)

        # get k smallest WITHOUT full sort
        k_indices = np.argpartition(distances, self.k)[:self.k]
        k_labels = self.y_train[k_indices]

        return np.bincount(k_labels).argmax()

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.array([self._predict_single(x) for x in X])

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)