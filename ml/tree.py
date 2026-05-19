import numpy as np


class DecisionTreeClassifierManual:
    """Decision Tree classifier tự code, dùng Gini impurity.

    Có hỗ trợ sample_weight để Random Forest có thể mô phỏng class_weight='balanced'.
    Không dùng scikit-learn.
    """

    def __init__(self, max_depth=8, min_samples_split=5, min_samples_leaf=2,
                 max_features="sqrt", n_thresholds=16, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.root = None
        self.n_features_ = None

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
        self.n_features_ = X.shape[1]
        self.root = self._build_tree(X, y, sample_weight, depth=0)
        return self

    def _gini(self, y, w):
        if len(y) == 0:
            return 0.0
        total_w = float(np.sum(w))
        if total_w <= 0:
            return 0.0
        p1 = float(np.sum(w[y == 1]) / total_w)
        p0 = 1.0 - p1
        return 1.0 - p0 ** 2 - p1 ** 2

    def _leaf(self, y, w):
        if len(y) == 0 or np.sum(w) <= 0:
            p1 = 0.0
        else:
            p1 = float(np.sum(w[y == 1]) / np.sum(w))
        pred = 1 if p1 >= 0.5 else 0
        return {
            "type": "leaf",
            "prediction": pred,
            "proba": p1,
            "n_samples": int(len(y)),
            "weighted_samples": float(np.sum(w)),
        }

    def _num_features_to_try(self):
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(self.n_features_)))
        if self.max_features == "log2":
            return max(1, int(np.log2(self.n_features_)))
        if self.max_features is None:
            return self.n_features_
        if isinstance(self.max_features, int):
            return max(1, min(self.max_features, self.n_features_))
        if isinstance(self.max_features, float):
            return max(1, min(int(self.max_features * self.n_features_), self.n_features_))
        return self.n_features_

    def _candidate_thresholds(self, values):
        unique = np.unique(values)
        if len(unique) <= 1:
            return []
        if len(unique) <= self.n_thresholds:
            return ((unique[:-1] + unique[1:]) / 2.0).tolist()
        qs = np.linspace(0.05, 0.95, self.n_thresholds)
        thresholds = np.quantile(values, qs)
        return np.unique(thresholds).tolist()

    def _best_split(self, X, y, w):
        n_samples, n_features = X.shape
        total_w = float(np.sum(w))
        parent_gini = self._gini(y, w)
        if parent_gini == 0 or total_w <= 0:
            return None, None, 0.0

        n_try = self._num_features_to_try()
        feature_indices = self.rng.choice(n_features, size=n_try, replace=False)

        best_feature = None
        best_threshold = None
        best_gain = 0.0

        for feature in feature_indices:
            values = X[:, feature]
            for threshold in self._candidate_thresholds(values):
                left_mask = values <= threshold
                right_mask = ~left_mask
                n_left = int(left_mask.sum())
                n_right = n_samples - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                w_left = w[left_mask]
                w_right = w[right_mask]
                wl = float(np.sum(w_left))
                wr = float(np.sum(w_right))
                if wl <= 0 or wr <= 0:
                    continue

                g_left = self._gini(y[left_mask], w_left)
                g_right = self._gini(y[right_mask], w_right)
                weighted_gini = (wl / total_w) * g_left + (wr / total_w) * g_right
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = float(threshold)

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, w, depth):
        max_depth_reached = self.max_depth is not None and depth >= self.max_depth
        if (
            max_depth_reached or
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1
        ):
            return self._leaf(y, w)

        feature, threshold, gain = self._best_split(X, y, w)
        if feature is None or gain <= 0:
            return self._leaf(y, w)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return {
            "type": "node",
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], w[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], w[right_mask], depth + 1),
        }

    def _predict_one_proba(self, x, node):
        while node["type"] != "leaf":
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["proba"]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p1 = np.array([self._predict_one_proba(row, self.root) for row in X])
        return np.vstack([1.0 - p1, p1]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
