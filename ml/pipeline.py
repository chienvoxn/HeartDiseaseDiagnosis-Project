import pickle
from .preprocessing import ManualPreprocessingPipeline
from .random_forest import RandomForestClassifierManual


class HeartDiseaseManualPipeline:
    """Pipeline đầy đủ: preprocessing tự code + Random Forest tự code."""

    def __init__(self, preprocessor=None, model=None):
        self.preprocessor = preprocessor or ManualPreprocessingPipeline()
        self.model = model or RandomForestClassifierManual()

    def fit(self, X, y):
        X_processed = self.preprocessor.fit_transform(X)
        self.model.fit(X_processed, y)
        return self

    def predict_proba(self, X):
        X_processed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_processed)

    def predict(self, X):
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
