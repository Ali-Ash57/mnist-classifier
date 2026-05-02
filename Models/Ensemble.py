import numpy as np

# -------------------------
# Decision Tree
# -------------------------
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=5, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.tree = None

    def fit(self, X, y):
        self.n_features = self.n_features or X.shape[1]
        self.tree = self._grow_tree(X, y)

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None

        feat_idxs = np.random.choice(X.shape[1], self.n_features, replace=False)

        for feat in feat_idxs:
            values = X[:, feat]
            n_samples = len(values)

            # ✅ FIXED: prevent sampling error
            size = min(10, n_samples)
            thresholds = np.random.choice(values, size=size, replace=False)

            for t in thresholds:
                left_mask = values <= t
                right_mask = values > t

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                gain = self._gini(y) - (
                    len(y_left)/len(y) * self._gini(y_left) +
                    len(y_right)/len(y) * self._gini(y_right)
                )

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat
                    split_thresh = t

        return split_idx, split_thresh

    def _grow_tree(self, X, y, depth=0):
        # stopping conditions
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return self._most_common(y)

        feat, thresh = self._best_split(X, y)

        if feat is None:
            return self._most_common(y)

        left_mask = X[:, feat] <= thresh
        right_mask = X[:, feat] > thresh

        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return (feat, thresh, left, right)

    def _most_common(self, y):
        return np.bincount(y).argmax()

    def _traverse(self, x, node):
        if not isinstance(node, tuple):
            return node

        feat, thresh, left, right = node

        if x[feat] <= thresh:
            return self._traverse(x, left)
        return self._traverse(x, right)

    def predict(self, X):
        return np.array([self._traverse(x, self.tree) for x in X])


# -------------------------
# Random Forest
# -------------------------
class RandomForest:
    def __init__(self, n_estimators=30, max_depth=10, min_samples_split=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape

        for _ in range(self.n_estimators):
            # bootstrap sampling
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[idxs], y[idxs]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=int(np.sqrt(n_features))
            )

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        return np.array([np.bincount(row).argmax() for row in tree_preds])

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)