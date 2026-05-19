import json

import matplotlib.pyplot as plt

from .config import DATA_PATH, MODEL_PATH, REPORT_PATH
from .data_utils import load_heart_data, stratified_train_test_split
from .pipeline import HeartDiseaseManualPipeline
from .random_forest import RandomForestClassifierManual
from .metrics import classification_metrics, print_classification_report


def plot_test_metrics(test_metrics, save_path):
    """Vẽ biểu đồ 4 metric trên tập test trước khi fine-tuning."""
    metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]
    metric_values = [
        test_metrics["accuracy"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(metric_names, metric_values)

    plt.ylim(0, 1)
    plt.title("Test Metrics Before Fine-tuning")
    plt.xlabel("Metric")
    plt.ylabel("Score")

    for bar, value in zip(bars, metric_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def main():
    # Load dữ liệu từ file CSV
    X, y = load_heart_data(DATA_PATH)

    # Chia dữ liệu thành train/test theo tỷ lệ class
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Khởi tạo Random Forest mặc định, chưa fine-tune
    model = RandomForestClassifierManual(
        n_estimators=120,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        n_thresholds=16,
        class_weight=None,
        bootstrap=True,
        random_state=42,
    )

    # Tạo pipeline gồm preprocessing + model
    pipeline = HeartDiseaseManualPipeline(model=model)

    # Huấn luyện pipeline trên tập train
    pipeline.fit(X_train, y_train)

    # Dự đoán trên tập train
    y_train_pred = pipeline.predict(X_train)

    # Dự đoán trên tập test
    y_test_pred = pipeline.predict(X_test)

    # Tính metrics
    train_metrics = classification_metrics(y_train, y_train_pred)
    test_metrics = classification_metrics(y_test, y_test_pred)

    # Tạo report để lưu ra file JSON
    report = {
        "model_status": "before_fine_tuning",
        "train": train_metrics,
        "test": test_metrics,
        "model": {
            "name": "Manual Random Forest",
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split,
            "min_samples_leaf": model.min_samples_leaf,
            "max_features": model.max_features,
            "n_thresholds": model.n_thresholds,
            "class_weight": model.class_weight,
            "bootstrap": model.bootstrap,
        },
        "features": pipeline.preprocessor.get_feature_names_out(),
    }

    # Tạo thư mục lưu model và report nếu chưa có
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Lưu pipeline đã train gồm preprocessing + model
    pipeline.save(MODEL_PATH)

    # Lưu report ra file JSON
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Vẽ và lưu biểu đồ 4 metric trên tập test
    test_metrics_plot_path = REPORT_PATH.parent / "test_metrics_before_finetune.png"
    plot_test_metrics(test_metrics, test_metrics_plot_path)

    # In đường dẫn lưu
    print("Model saved to:", MODEL_PATH)
    print("Report saved to:", REPORT_PATH)
    print("Test metrics plot saved to:", test_metrics_plot_path)

    # In kết quả đánh giá trên tập train
    print("\nTrain metrics:")
    print_classification_report(y_train, y_train_pred)

    # In kết quả đánh giá trên tập test
    print("\nTest metrics:")
    print_classification_report(y_test, y_test_pred)


if __name__ == "__main__":
    main()
