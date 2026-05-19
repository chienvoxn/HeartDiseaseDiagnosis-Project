import numpy as np
import pandas as pd
from .config import DROP_COLUMNS, TARGET_COLUMN


def load_heart_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.copy()

    # Binary classification: num = 0 -> 0, num > 0 -> 1
    df[TARGET_COLUMN] = df["num"].apply(lambda x: 0 if x == 0 else 1)

    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET_COLUMN].astype(int).to_numpy()
    return X, y


def stratified_train_test_split(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    train_indices = []
    test_indices = []

    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        test_indices.extend(idx[:n_test])
        train_indices.extend(idx[n_test:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return X.iloc[train_indices], X.iloc[test_indices], y[train_indices], y[test_indices]
