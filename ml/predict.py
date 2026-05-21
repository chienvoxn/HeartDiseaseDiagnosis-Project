from .config import MODEL_PATH
from .pipeline import HeartDiseaseManualPipeline

_model = None


def load_model():
    global _model
    if _model is None:
        _model = HeartDiseaseManualPipeline.load(MODEL_PATH)
    return _model


def predict_heart_disease(input_data):
    """input_data là dict có các key đúng tên feature.

    Return:
        prediction: 0 hoặc 1
        probability: xác suất class 1
    """
    model = load_model()
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])
    return prediction, probability


if __name__ == "__main__":
    sample = {
        "age": 55,
        "sex": "Male",
        "cp": "asymptomatic",
        "trestbps": 140,
        "chol": 250,
        "fbs": False,
        "restecg": "normal",
        "thalch": 150,
        "exang": True,
        "oldpeak": 1.5,
        "slope": "flat",
        "ca": 0,
        "thal": "normal",
    }
    print(predict_heart_disease(sample))
