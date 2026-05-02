import numpy as np
from logistic_regression import LogisticRegression


class LogisticRegressionOVR:
    def __init__(self, learning_rate=0.01, epochs=1000, lambda_=0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_ = lambda_
        self.models = []
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = []

        for c in self.classes:
            y_binary = (y == c).astype(int)

            model = LogisticRegression(
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                lambda_=self.lambda_
            )

            model.fit(X, y_binary)
            self.models.append(model)

    def predict(self, X):
        probs = []

        for model in self.models:
            p = model.predict_proba(X)
            probs.append(p)

        probs = np.array(probs).T
        return self.classes[np.argmax(probs, axis=1)]

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)