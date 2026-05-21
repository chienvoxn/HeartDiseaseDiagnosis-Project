import json
import sys
from pathlib import Path
from django.contrib.auth.decorators import login_required
import pandas as pd
from django.shortcuts import render
from django.conf import settings
from datetime import datetime
from .forms import HeartDiseaseForm, CSVUploadForm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ml.predict import predict_heart_disease


# RISK_THRESHOLD = 0.40


FEATURE_NAME_MAP = {
    "num_age": "Tuổi",
    "num_trestbps": "Huyết áp khi nghỉ",
    "num_chol": "Cholesterol",
    "num_thalch": "Nhịp tim tối đa",
    "num_oldpeak": "Chỉ số ST chênh xuống",

    "cat_sex_Male": "Giới tính: Nam",
    "cat_sex_Female": "Giới tính: Nữ",

    "cat_cp_asymptomatic": "Đau ngực: Không có triệu chứng",
    "cat_cp_atypical angina": "Đau ngực: Không điển hình",
    "cat_cp_non-anginal": "Đau ngực không do tim",
    "cat_cp_typical angina": "Đau thắt ngực điển hình",

    "cat_exang_True": "Có đau thắt ngực khi vận động",
    "cat_exang_False": "Không đau thắt ngực khi vận động",

    "cat_thal_normal": "Thalassemia: Bình thường",
    "cat_thal_fixed defect": "Thalassemia: Khiếm khuyết cố định",
    "cat_thal_reversable defect": "Thalassemia: Khiếm khuyết có thể đảo ngược",

    "cat_slope_upsloping": "Độ dốc ST: Dốc lên",
    "cat_slope_flat": "Độ dốc ST: Phẳng",
    "cat_slope_downsloping": "Độ dốc ST: Dốc xuống",

    "cat_fbs_True": "Đường huyết lúc đói > 120 mg/dl",
    "cat_fbs_False": "Đường huyết lúc đói không cao",

    "cat_restecg_normal": "Điện tâm đồ: Bình thường",
    "cat_restecg_st-t abnormality": "Điện tâm đồ: Bất thường ST-T",
    "cat_restecg_lv hypertrophy": "Điện tâm đồ: Phì đại thất trái",
}


REQUIRED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalch", "exang", "oldpeak",
    "slope", "ca", "thal"
]


def _clean_form_data(cleaned_data):
    data = dict(cleaned_data)

    data["fbs"] = True if data["fbs"] == "True" else False
    data["exang"] = True if data["exang"] == "True" else False

    return data


def _build_simple_explanation(input_data):
    increase = []
    decrease = []

    if input_data["cp"] == "asymptomatic":
        increase.append("Đau ngực không có triệu chứng là yếu tố thường liên quan đến nguy cơ cao.")
    else:
        decrease.append("Loại đau ngực không thuộc nhóm không triệu chứng nên nguy cơ có thể thấp hơn.")

    if float(input_data["oldpeak"]) >= 2:
        increase.append("Chỉ số ST chênh xuống cao.")
    else:
        decrease.append("Chỉ số ST chênh xuống không quá cao.")

    if input_data["exang"] is True:
        increase.append("Có đau thắt ngực khi vận động.")
    else:
        decrease.append("Không đau thắt ngực khi vận động.")

    if float(input_data["thalch"]) < 130:
        increase.append("Nhịp tim tối đa đạt được thấp.")
    else:
        decrease.append("Nhịp tim tối đa đạt được tương đối tốt.")

    if int(input_data["age"]) >= 55:
        increase.append("Tuổi thuộc nhóm có nguy cơ tim mạch cao hơn.")

    if str(input_data["ca"]) not in ["0", "0.0"]:
        increase.append("Có mạch máu chính được phát hiện bất thường.")
    else:
        decrease.append("Số mạch máu chính được phát hiện là 0.")

    return {
        "increase": increase[:4],
        "decrease": decrease[:4],
    }


def _predict_one_patient(input_data):
    prediction, disease_probability = predict_heart_disease(input_data)

    probability_percent = round(disease_probability * 100, 2)

    if prediction == 1:
        result = "Có nguy cơ mắc bệnh tim"
        risk_label = "danger"
    else:
        result = "Không có nguy cơ mắc bệnh tim"
        risk_label = "safe"

    return {
        "prediction": prediction,
        "probability": probability_percent,
        "result": result,
        "risk_label": risk_label,
    }
def _read_uploaded_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def _read_csv_predictions(test_file, answer_file=None):
    df = _read_uploaded_file(test_file)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError("File test thiếu các cột: " + ", ".join(missing_columns))

    if "name" not in df.columns:
        raise ValueError("File test cần có cột name để định danh bệnh nhân.")

    answer_df = None
    has_answer = answer_file is not None

    if has_answer:
        answer_df = _read_uploaded_file(answer_file)

        if "name" not in answer_df.columns or "target" not in answer_df.columns:
            raise ValueError("File đáp án cần có 2 cột: name và target.")

        answer_df = answer_df[["name", "target"]]

    batch_results = []
    predictions = []

    for index, row in df.iterrows():
        input_data = row[REQUIRED_COLUMNS].to_dict()

        # Map dữ liệu số sang dạng model đã train
        if str(input_data["sex"]) in ["0", "0.0"]:
            input_data["sex"] = "Female"
        elif str(input_data["sex"]) in ["1", "1.0"]:
            input_data["sex"] = "Male"

        cp_map = {
            "0": "typical angina", "0.0": "typical angina",
            "1": "atypical angina", "1.0": "atypical angina",
            "2": "non-anginal", "2.0": "non-anginal",
            "3": "asymptomatic", "3.0": "asymptomatic",
        }
        input_data["cp"] = cp_map.get(str(input_data["cp"]), input_data["cp"])

        input_data["fbs"] = str(input_data["fbs"]).lower() in ["true", "1", "1.0"]

        restecg_map = {
            "0": "normal", "0.0": "normal",
            "1": "st-t abnormality", "1.0": "st-t abnormality",
            "2": "lv hypertrophy", "2.0": "lv hypertrophy",
        }
        input_data["restecg"] = restecg_map.get(str(input_data["restecg"]), input_data["restecg"])

        input_data["exang"] = str(input_data["exang"]).lower() in ["true", "1", "1.0"]

        slope_map = {
            "0": "upsloping", "0.0": "upsloping",
            "1": "flat", "1.0": "flat",
            "2": "downsloping", "2.0": "downsloping",
        }
        input_data["slope"] = slope_map.get(str(input_data["slope"]), input_data["slope"])

        thal_map = {
            "0": "normal", "0.0": "normal",
            "1": "normal", "1.0": "normal",
            "2": "fixed defect", "2.0": "fixed defect",
            "3": "reversable defect", "3.0": "reversable defect",
        }
        input_data["thal"] = thal_map.get(str(input_data["thal"]), input_data["thal"])

        input_data["ca"] = str(input_data["ca"]).replace(".0", "")

        predict_result = _predict_one_patient(input_data)

        raw_prediction = int(predict_result["prediction"])

        # Giữ logic đã test đúng với file heart trước đó
        prediction = 1 - raw_prediction

        result_text = "Có nguy cơ mắc bệnh tim" if prediction == 1 else "Không có nguy cơ mắc bệnh tim"
        risk_label = "danger" if prediction == 1 else "safe"

        patient_name = row["name"]

        actual_target = None
        match_result = ""

        if has_answer:
            matched = answer_df[answer_df["name"] == patient_name]

            if not matched.empty:
                actual_target = int(matched.iloc[0]["target"])
                match_result = "Đúng" if prediction == actual_target else "Sai"
            else:
                match_result = "Không tìm thấy đáp án"

        predictions.append({
            "name": patient_name,
            "predicted_label": prediction,
            "predicted_result": result_text,
            "actual_target": actual_target,
            "match": match_result,
        })

        batch_results.append({
            "stt": index + 1,
            "name": patient_name,
            "age": input_data.get("age"),
            "sex": "Nam" if input_data.get("sex") == "Male" else "Nữ",
            "prediction": prediction,
            "actual_target": actual_target,
            "match": match_result,
            "result": result_text,
            "risk_label": risk_label,
        })

    prediction_df = pd.DataFrame(predictions)
    output_df = pd.concat([df.reset_index(drop=True), prediction_df.drop(columns=["name"])], axis=1)

    output_dir = settings.MEDIA_ROOT / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"heart_prediction_results_{timestamp}.xlsx"
    output_path = output_dir / output_filename

    summary_data = {
        "Tổng số bệnh nhân": [len(output_df)],
        "Số ca có nguy cơ": [int((output_df["predicted_label"] == 1).sum())],
        "Số ca không có nguy cơ": [int((output_df["predicted_label"] == 0).sum())],
    }

    if has_answer:
        valid_compare = output_df[output_df["match"].isin(["Đúng", "Sai"])]
        total = len(valid_compare)
        correct = int((valid_compare["match"] == "Đúng").sum())

        if total > 0:
            summary_data["Số dòng so sánh được"] = [total]
            summary_data["Số dòng dự đoán đúng"] = [correct]
            summary_data["Số dòng dự đoán sai"] = [total - correct]
            summary_data["Accuracy (%)"] = [round(correct / total * 100, 2)]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        output_df.to_excel(writer, index=False, sheet_name="Predictions")
        pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name="Summary")

    output_url = settings.MEDIA_URL + f"exports/{output_filename}"

    accuracy_info = None

    if has_answer:
        valid_compare = output_df[output_df["match"].isin(["Đúng", "Sai"])]
        total = len(valid_compare)
        correct = int((valid_compare["match"] == "Đúng").sum())

        if total > 0:
            accuracy_info = {
                "total": total,
                "correct": correct,
                "wrong": total - correct,
                "accuracy": round(correct / total * 100, 2),
            }

    return batch_results, output_url, accuracy_info
def diagnosis_view(request):
    form = HeartDiseaseForm()
    upload_form = CSVUploadForm()
    

    result = None
    probability = None
    risk_label = None
    shap_explanation = None

    batch_results = []
    upload_error = None
    export_file_url = None

    if request.method == "POST":
        form_type = request.POST.get("form_type")

        if form_type == "manual":
            form = HeartDiseaseForm(request.POST)

            if form.is_valid():
                input_data = _clean_form_data(form.cleaned_data)

                predict_result = _predict_one_patient(input_data)

                result = predict_result["result"]
                probability = predict_result["probability"]
                risk_label = predict_result["risk_label"]
                shap_explanation = _build_simple_explanation(input_data)

        elif form_type == "csv":
            upload_form = CSVUploadForm(request.POST, request.FILES)

            if upload_form.is_valid():
                try:
                    csv_file = upload_form.cleaned_data["csv_file"]
                    batch_results, export_file_url = _read_csv_predictions(csv_file)
                except Exception as e:
                    upload_error = str(e)

    return render(request, "diagnosis/index.html", {
        "form": form,
        "upload_form": upload_form,

        "result": result,
        "probability": probability,
        "risk_label": risk_label,
        "shap_explanation": shap_explanation,

        "batch_results": batch_results,
        "upload_error": upload_error,
        "export_file_url": export_file_url,
    })


def explanation_view(request):
    report_path = PROJECT_ROOT / "ml" / "reports" / "shap_report.json"
    top_features = []
    shap_available = report_path.exists()

    if shap_available:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        raw_features = report.get("top_features", [])[:8]

        for item in raw_features:
            feature_name = item.get("feature", "")
            top_features.append({
                "feature": FEATURE_NAME_MAP.get(feature_name, feature_name),
                "mean_abs_shap": item.get("mean_abs_shap", 0),
            })

    return render(request, "diagnosis/explanation.html", {
        "shap_available": shap_available,
        "top_features": top_features,
    })
@login_required
def dashboard_view(request):
    data_dir = PROJECT_ROOT / "data"

    dataset_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.xlsx"))

    if not dataset_files:
        return render(request, "diagnosis/dashboard.html", {
            "total_patients": 0,
            "disease_count": 0,
            "healthy_count": 0,
            "columns": [],
            "preview_data": [],
            "dataset_name": "Chưa có file dữ liệu",
            "dataset_error": "Không tìm thấy file .csv hoặc .xlsx trong thư mục data/",
        })

    dataset_path = dataset_files[0]

    if dataset_path.suffix == ".csv":
        df = pd.read_csv(dataset_path)
    else:
        df = pd.read_excel(dataset_path)

    total_patients = len(df)

    target_col = None

    if "target" in df.columns:
        target_col = "target"
    elif "num" in df.columns:
        target_col = "num"

    if target_col:
        if target_col == "num":
            disease_count = int((df[target_col] > 0).sum())
            healthy_count = int((df[target_col] == 0).sum())
        else:
            disease_count = int((df[target_col] == 1).sum())
            healthy_count = int((df[target_col] == 0).sum())
    else:
        disease_count = 0
        healthy_count = 0

    preview_data = df.head(10).to_dict(orient="records")
    columns = df.columns.tolist()

    return render(request, "diagnosis/dashboard.html", {
        "total_patients": total_patients,
        "disease_count": disease_count,
        "healthy_count": healthy_count,
        "columns": columns,
        "preview_data": preview_data,
        "dataset_name": dataset_path.name,
        "dataset_error": None,
    })

@login_required
def predict_view(request):
    form = HeartDiseaseForm()
    upload_form = CSVUploadForm()

    result = None
    risk_label = None
    shap_explanation = None

    batch_results = []
    upload_error = None
    export_file_url = None
    accuracy_info = None

    if request.method == "POST":
        form_type = request.POST.get("form_type")

        if form_type == "manual":
            form = HeartDiseaseForm(request.POST)

            if form.is_valid():
                input_data = _clean_form_data(form.cleaned_data)
                predict_result = _predict_one_patient(input_data)

                result = predict_result["result"]
                risk_label = predict_result["risk_label"]
                shap_explanation = _build_simple_explanation(input_data)

        elif form_type == "csv":
            upload_form = CSVUploadForm(request.POST, request.FILES)

            if upload_form.is_valid():
                try:
                    test_file = upload_form.cleaned_data["test_file"]
                    answer_file = upload_form.cleaned_data.get("answer_file")

                    batch_results, export_file_url, accuracy_info = _read_csv_predictions(
                        test_file,
                        answer_file
                    )
                    request.session["latest_batch_results"] = batch_results
                    request.session["latest_export_file_url"] = export_file_url

                except Exception as e:
                    upload_error = str(e)

    return render(request, "diagnosis/predict.html", {
        "form": form,
        "upload_form": upload_form,

        "result": result,
        "risk_label": risk_label,
        "shap_explanation": shap_explanation,

        "batch_results": batch_results,
        "upload_error": upload_error,
        "export_file_url": export_file_url,
        "accuracy_info": accuracy_info,
    })

@login_required
def evaluation_view(request):
    batch_results = request.session.get("latest_batch_results", [])
    export_file_url = request.session.get("latest_export_file_url")

    if not batch_results:
        return render(request, "diagnosis/evaluation.html", {
            "has_result": False,
            "message": "Chưa có dữ liệu đánh giá. Vui lòng vào mục Dự đoán bệnh tim và upload file trước.",
        })

    y_true = []
    y_pred = []

    for item in batch_results:
        actual_target = item.get("actual_target")
        prediction = item.get("prediction")

        if actual_target is not None and actual_target != "":
            y_true.append(int(actual_target))
            y_pred.append(int(prediction))

    if not y_true:
        return render(request, "diagnosis/evaluation.html", {
            "has_result": False,
            "message": "Kết quả dự đoán hiện tại chưa có target thật để đánh giá. Hãy upload thêm file đáp án ở mục Dự đoán bệnh tim.",
        })

    acc = round(accuracy_score(y_true, y_pred) * 100, 2)
    prec = round(precision_score(y_true, y_pred, zero_division=0) * 100, 2)
    rec = round(recall_score(y_true, y_pred, zero_division=0) * 100, 2)
    f1 = round(f1_score(y_true, y_pred, zero_division=0) * 100, 2)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    correct = sum(1 for a, p in zip(y_true, y_pred) if a == p)
    wrong = len(y_true) - correct

    return render(request, "diagnosis/evaluation.html", {
        "has_result": True,

        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,

        "total": len(y_true),
        "correct": correct,
        "wrong": wrong,

        "export_file_url": export_file_url,
    })
