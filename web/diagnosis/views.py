import json
import sys
from pathlib import Path

from django.shortcuts import render
from .forms import HeartDiseaseForm

# Cho phép Django import module ml ở thư mục cha của web/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ml.predict import predict_heart_disease


def _clean_form_data(cleaned_data):
    data = dict(cleaned_data)
    data["fbs"] = True if data["fbs"] == "True" else False
    data["exang"] = True if data["exang"] == "True" else False
    data["ca"] = int(data["ca"])
    return data


def diagnosis_view(request):
    result = None
    probability = None
    risk_label = None

    if request.method == "POST":
        form = HeartDiseaseForm(request.POST)
        if form.is_valid():
            input_data = _clean_form_data(form.cleaned_data)
            prediction, probability = predict_heart_disease(input_data)
            probability_percent = round(probability * 100, 2)

            if prediction == 1:
                result = "Có nguy cơ mắc bệnh tim"
                risk_label = "danger"
            else:
                result = "Không có nguy cơ mắc bệnh tim"
                risk_label = "safe"

            probability = probability_percent
    else:
        form = HeartDiseaseForm()

    return render(request, "diagnosis/index.html", {
        "form": form,
        "result": result,
        "probability": probability,
        "risk_label": risk_label,
    })


def explanation_view(request):
    """Trang hiển thị SHAP global explanation đã được tạo bằng ml/shap_explain.py."""
    report_path = PROJECT_ROOT / "ml" / "reports" / "shap_report.json"
    top_features = []
    shap_available = report_path.exists()

    if shap_available:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        top_features = report.get("top_features", [])[:10]

    return render(request, "diagnosis/explanation.html", {
        "shap_available": shap_available,
        "top_features": top_features,
    })
