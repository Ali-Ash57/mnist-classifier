import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, lambda_=0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, y_true, y_pred):
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        m = y_true.shape[0]

        # Binary cross-entropy
        loss = -np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        ) / m

        # L2 regularization term (only weights)
        l2_term = (self.lambda_ / (2 * m)) * np.sum(self.weights ** 2)

        return loss + l2_term

    def fit(self, X, y):
        m, n = X.shape

        self.weights = np.zeros(n)
        self.bias = 0.0

        for epoch in range(self.epochs):
            # forward
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)

            # loss
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # gradients
            dw = (np.dot(X.T, (y_pred - y)) / m) + (self.lambda_ / m) * self.weights
            db = np.sum(y_pred - y) / m

            # update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)