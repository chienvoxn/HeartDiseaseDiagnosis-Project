import numpy as np
from .tree import DecisionTreeClassifierManual


class RandomForestClassifierManual:
    """Random Forest tự code.

    Ý tưởng:
    - Bootstrap sampling để tạo nhiều tập train con.
    - Mỗi cây là một Decision Tree tự code.
    - Mỗi node chỉ thử một tập con feature ngẫu nhiên.
    - Classification dùng probability averaging / majority voting.
    - Hỗ trợ class_weight=None hoặc 'balanced' để giống notebook sklearn hơn.
    """

    def __init__(self, n_estimators=120, max_depth=8, min_samples_split=5,
                 min_samples_leaf=2, max_features="sqrt", n_thresholds=16,
                 class_weight=None, bootstrap=True, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.class_weight = class_weight
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.rng = np.random.default_rng(random_state)

    def _compute_sample_weight(self, y):
        y = np.asarray(y, dtype=int)
        weights = np.ones(len(y), dtype=float)
        if self.class_weight == "balanced":
            classes, counts = np.unique(y, return_counts=True)
            n_samples = len(y)
            n_classes = len(classes)
            weight_map = {cls: n_samples / (n_classes * cnt) for cls, cnt in zip(classes, counts)}
            weights = np.array([weight_map[label] for label in y], dtype=float)
        return weights

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_samples = X.shape[0]
        base_weight = self._compute_sample_weight(y)
        self.trees = []
        self.rng = np.random.default_rng(self.random_state)

        for i in range(self.n_estimators):
            if self.bootstrap:
                indices = self.rng.integers(0, n_samples, size=n_samples)
            else:
                indices = np.arange(n_samples)

            tree = DecisionTreeClassifierManual(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_thresholds=self.n_thresholds,
                random_state=self.random_state + i if self.random_state is not None else None,
            )
            tree.fit(X[indices], y[indices], sample_weight=base_weight[indices])
            self.trees.append(tree)

        return self

    def predict_proba(self, X):
        if not self.trees:
            raise RuntimeError("Model chưa được train.")
        probs = np.array([tree.predict_proba(X)[:, 1] for tree in self.trees])
        p1 = probs.mean(axis=0)
        return np.vstack([1.0 - p1, p1]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
