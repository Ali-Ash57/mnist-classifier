import numpy as np

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.mean = np.zeros((n_classes, n_features), dtype=np.float32)
        self.var = np.zeros((n_classes, n_features), dtype=np.float32)
        self.priors = np.zeros(n_classes, dtype=np.float32)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-9
            self.priors[idx] = X_c.shape[0] / X.shape[0]

    def _gaussian_log_pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]

        return -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(((x - mean) ** 2) / var)

    def _predict_single(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            likelihood = self._gaussian_log_pdf(idx, x)
            posterior = prior + likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)