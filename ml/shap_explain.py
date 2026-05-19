"""SHAP explanation cho Manual Random Forest.

Phần train/preprocessing/model trong project KHÔNG dùng scikit-learn.
SHAP được dùng như thư viện XAI ở bước giải thích mô hình sau khi train.

Vì model là Random Forest tự code, ta dùng shap.KernelExplainer với hàm
predict_proba của manual model. Script này tạo biểu đồ SHAP và bảng feature
importance dựa trên mean(|SHAP value|).

Chạy:
    python -m ml.shap_explain

Output:
    ml/reports/shap_summary_bar.png
    ml/reports/shap_summary_dot.png
    ml/reports/shap_feature_importance.csv
    ml/reports/shap_report.json

Ngoài ra, script cũng copy ảnh SHAP sang:
    web/diagnosis/static/diagnosis/images/
để Django có thể hiển thị ở trang Model Explanation.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import DATA_PATH, MODEL_PATH, REPORT_PATH
from .data_utils import load_heart_data, stratified_train_test_split
from .pipeline import HeartDiseaseManualPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_IMAGE_DIR = PROJECT_ROOT / "web" / "diagnosis" / "static" / "diagnosis" / "images"


def make_prediction_function(model):
    """Hàm trả xác suất class 1 cho SHAP KernelExplainer."""
    def predict_class_1(X_processed):
        return model.predict_proba(X_processed)[:, 1]
    return predict_class_1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background-size", type=int, default=30)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--nsamples", type=int, default=80)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "Bạn cần cài SHAP trước: pip install shap matplotlib"
        ) from exc

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model tại {MODEL_PATH}. Hãy chạy: python -m ml.grid_search_random_forest --fast --cv 3"
        )

    X, y = load_heart_data(DATA_PATH)
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_size=0.2, random_state=args.random_state
    )

    pipeline = HeartDiseaseManualPipeline.load(MODEL_PATH)
    preprocessor = pipeline.preprocessor
    rf_model = pipeline.model

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    rng = np.random.default_rng(args.random_state)
    bg_size = min(args.background_size, len(X_train_processed))
    sample_size = min(args.sample_size, len(X_test_processed))

    bg_idx = rng.choice(len(X_train_processed), size=bg_size, replace=False)
    sample_idx = rng.choice(len(X_test_processed), size=sample_size, replace=False)

    background = X_train_processed[bg_idx]
    X_explain = X_test_processed[sample_idx]

    predict_fn = make_prediction_function(rf_model)
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_explain, nsamples=args.nsamples)

    reports_dir = REPORT_PATH.parent
    reports_dir.mkdir(parents=True, exist_ok=True)
    WEB_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    X_explain_df = pd.DataFrame(X_explain, columns=feature_names)

    bar_path = reports_dir / "shap_summary_bar.png"
    dot_path = reports_dir / "shap_summary_dot.png"
    importance_csv_path = reports_dir / "shap_feature_importance.csv"

    plt.figure()
    shap.summary_plot(shap_values, X_explain_df, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_explain_df, show=False)
    plt.tight_layout()
    plt.savefig(dot_path, dpi=160, bbox_inches="tight")
    plt.close()

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    shap_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)
    shap_importance.to_csv(importance_csv_path, index=False)

    # Copy ảnh cho Django hiển thị ở trang /explanation/
    web_bar_path = WEB_IMAGE_DIR / "shap_summary_bar.png"
    web_dot_path = WEB_IMAGE_DIR / "shap_summary_dot.png"
    shutil.copyfile(bar_path, web_bar_path)
    shutil.copyfile(dot_path, web_dot_path)

    meta = {
        "method": "SHAP KernelExplainer",
        "note": "SHAP is used only for model explanation. Manual preprocessing and Manual Random Forest do not use scikit-learn.",
        "model": "Manual Random Forest",
        "background_size": int(bg_size),
        "sample_size": int(sample_size),
        "nsamples": int(args.nsamples),
        "top_features": shap_importance.head(15).to_dict(orient="records"),
        "outputs": {
            "bar_plot": str(bar_path),
            "dot_plot": str(dot_path),
            "importance_csv": str(importance_csv_path),
            "django_bar_plot": str(web_bar_path),
            "django_dot_plot": str(web_dot_path),
        }
    }
    with open(reports_dir / "shap_report.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved SHAP bar plot to:", bar_path)
    print("Saved SHAP dot plot to:", dot_path)
    print("Saved SHAP feature importance CSV to:", importance_csv_path)
    print("Copied SHAP images to Django static folder:", WEB_IMAGE_DIR)


if __name__ == "__main__":
    main()
