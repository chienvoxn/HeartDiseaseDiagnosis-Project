"""Manual Grid Search cho Random Forest tự code.

Không dùng scikit-learn.

Chạy:
    python -m ml.grid_search_random_forest

Mục tiêu:
- Grid Search: thử toàn bộ tổ hợp hyperparameter.
- Cross-validation: dùng Stratified K-Fold tự code trên training set.
- Chọn model cân bằng 4 metric, trong đó Recall được ưu tiên nhẹ.
- Train lại final Random Forest trên toàn bộ training set.
- Evaluate trên test set bằng 4 metric.
- Vẽ biểu đồ 4 metric trên tập test sau fine-tuning.
- Lưu model để Django sử dụng.
"""

import argparse
import itertools
import json
import time

import matplotlib.pyplot as plt
import numpy as np

from .config import DATA_PATH, MODEL_PATH, REPORT_PATH
from .data_utils import load_heart_data, stratified_train_test_split
from .pipeline import HeartDiseaseManualPipeline
from .random_forest import RandomForestClassifierManual
from .metrics import classification_metrics


def stratified_kfold_indices(y, k=5, random_state=42):
    """Tạo Stratified K-Fold để giữ tỷ lệ class giữa các fold."""
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    folds = [[] for _ in range(k)]

    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        split_idx = np.array_split(idx, k)

        for fold_id, part in enumerate(split_idx):
            folds[fold_id].extend(part.tolist())

    all_indices = np.arange(len(y))
    result = []

    for fold_id in range(k):
        val_idx = np.array(folds[fold_id], dtype=int)
        train_idx = np.setdiff1d(all_indices, val_idx)

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)

        result.append((train_idx, val_idx))

    return result


def build_param_grid(fast=False):
    """Tạo danh sách hyperparameter cần thử.

    fast=True:
        Grid nhỏ để test nhanh.

    fast=False:
        Grid lớn hơn để fine-tune chính thức.

    Mục tiêu grid này:
    - Giảm overfitting bằng cách giới hạn độ sâu cây.
    - Tăng min_samples_split và min_samples_leaf.
    - Giữ model cân bằng giữa Accuracy, Precision, Recall và F1-score.
    """

    if fast:
        grid = {
            "n_estimators": [120],
            "max_depth": [4, 5, 6, 7],
            "min_samples_split": [10, 15],
            "min_samples_leaf": [4, 6],
            "max_features": ["sqrt"],
            "class_weight": [None],
            "n_thresholds": [12],
        }
    else:
        grid = {
            "n_estimators": [120, 200],
            "max_depth": [4, 5, 6, 7],
            "min_samples_split": [10, 15, 20],
            "min_samples_leaf": [4, 6, 8],
            "max_features": ["sqrt"],
            "class_weight": [None],
            "n_thresholds": [12, 20],
        }

    keys = list(grid.keys())

    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def evaluate_params_cv(X_train, y_train, params, cv=5, random_state=42):
    """Đánh giá một bộ hyperparameter bằng Stratified K-Fold CV."""
    y_train_arr = np.asarray(y_train)

    folds = stratified_kfold_indices(y_train_arr, k=cv, random_state=random_state)
    fold_metrics = []

    for fold_id, (tr_idx, val_idx) in enumerate(folds, start=1):
        X_tr = X_train.iloc[tr_idx]
        y_tr = y_train_arr[tr_idx]

        X_val = X_train.iloc[val_idx]
        y_val = y_train_arr[val_idx]

        model = RandomForestClassifierManual(
            **params,
            bootstrap=True,
            random_state=random_state + fold_id,
        )

        # Mỗi fold có pipeline riêng để fit preprocessing trên train fold,
        # sau đó transform validation fold. Cách này tránh data leakage.
        pipeline = HeartDiseaseManualPipeline(model=model)
        pipeline.fit(X_tr, y_tr)

        y_pred = pipeline.predict(X_val)
        metrics = classification_metrics(y_val, y_pred)
        fold_metrics.append(metrics)

    avg = {
        "accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "precision": float(np.mean([m["precision"] for m in fold_metrics])),
        "recall": float(np.mean([m["recall"] for m in fold_metrics])),
        "f1": float(np.mean([m["f1"] for m in fold_metrics])),
    }

    return avg, fold_metrics


def selection_score(metrics):
    """Điểm chọn model.

    Ưu tiên Recall và Accuracy.
    Recall quan trọng vì bài toán bệnh tim cần hạn chế bỏ sót người có bệnh.
    Accuracy được dùng để đảm bảo mô hình vẫn dự đoán đúng tổng thể tốt.
    Precision và F1-score vẫn được xét nhẹ để tránh mô hình quá lệch.
    """

    return (
        0.35 * metrics["recall"]
        + 0.35 * metrics["accuracy"]
        + 0.15 * metrics["precision"]
        + 0.15 * metrics["f1"]
    )


def grid_search(X_train, y_train, cv=5, random_state=42, fast=False):
    """Chạy Grid Search thủ công."""
    params_list = list(build_param_grid(fast=fast))
    total = len(params_list)
    results = []

    print(f"Start manual GridSearchCV | candidates={total} | cv={cv}")

    for i, params in enumerate(params_list, start=1):
        start = time.time()

        avg_metrics, fold_metrics = evaluate_params_cv(
            X_train,
            y_train,
            params,
            cv=cv,
            random_state=random_state + i * 10,
        )

        elapsed = time.time() - start

        row = {
            "iteration": i,
            "params": params,
            "cv_metrics": avg_metrics,
            "fold_metrics": fold_metrics,
            "elapsed_seconds": round(elapsed, 2),
        }

        results.append(row)

        print(
            f"[{i}/{total}] "
            f"Recall={avg_metrics['recall']:.4f} | "
            f"F1={avg_metrics['f1']:.4f} | "
            f"Acc={avg_metrics['accuracy']:.4f} | "
            f"Params={params} | {elapsed:.1f}s"
        )

    results_sorted = sorted(
        results,
        key=lambda r: selection_score(r["cv_metrics"]),
        reverse=True,
    )

    return results_sorted[0], results_sorted


def print_section_title(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_params(params):
    print("\nBest Hyperparameters:")
    print("-" * 60)

    for key, value in params.items():
        print(f"{key:<25}: {value}")


def print_metrics(title, metrics):
    print(f"\n{title}")
    print("-" * 60)
    print(f"{'Accuracy':<15}: {metrics['accuracy']:.4f}")
    print(f"{'Precision':<15}: {metrics['precision']:.4f}")
    print(f"{'Recall':<15}: {metrics['recall']:.4f}")
    print(f"{'F1-score':<15}: {metrics['f1']:.4f}")


def print_confusion_matrix(cm):
    tn, fp = cm[0]
    fn, tp = cm[1]

    print("\nConfusion Matrix:")
    print("-" * 60)
    print(f"{'':<18}{'Predicted 0':<18}{'Predicted 1':<18}")
    print(f"{'':<18}{'Không bệnh':<18}{'Có bệnh':<18}")
    print(f"{'Actual 0':<18}{tn:<18}{fp:<18}")
    print(f"{'Actual 1':<18}{fn:<18}{tp:<18}")

    print("\nGiải thích:")
    print(f"TN = {tn}: Thực tế không bệnh, mô hình dự đoán không bệnh")
    print(f"FP = {fp}: Thực tế không bệnh, mô hình dự đoán có bệnh")
    print(f"FN = {fn}: Thực tế có bệnh, mô hình dự đoán không bệnh")
    print(f"TP = {tp}: Thực tế có bệnh, mô hình dự đoán có bệnh")


def plot_test_metrics(test_metrics, save_path):
    """Vẽ và lưu biểu đồ 4 metric trên tập test sau khi fine-tuning."""
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
    plt.title("Final Test Metrics After Fine-tuning")
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
    plt.close()


def print_final_report(best, report, model_path, report_path, test_plot_path=None):
    """In kết quả cuối cùng sau Grid Search cho đẹp."""
    print_section_title("GRID SEARCH RESULT")

    print_params(best["params"])

    print_metrics("Best Cross-Validation Metrics", best["cv_metrics"])

    print_metrics("Train Metrics", report["train"])

    print_metrics("Final Test Metrics", report["test"])

    print_confusion_matrix(report["test"]["confusion_matrix"])

    print_section_title("SAVED FILES")
    print("Saved model to:", model_path)
    print("Saved grid search report to:", report_path)

    if test_plot_path is not None:
        print("Saved test metrics plot to:", test_plot_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=int, default=5, help="Số fold cross-validation")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Dùng grid nhỏ để test nhanh",
    )

    args = parser.parse_args()

    # Load dữ liệu đã được xử lý cơ bản trong data_utils
    X, y = load_heart_data(DATA_PATH)

    # Chia train/test theo tỷ lệ class
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.random_state,
    )

    # Grid Search trên training set
    best, all_results = grid_search(
        X_train,
        y_train,
        cv=args.cv,
        random_state=args.random_state,
        fast=args.fast,
    )

    # Train lại final model trên toàn bộ training set bằng best params
    final_rf = RandomForestClassifierManual(
        **best["params"],
        bootstrap=True,
        random_state=args.random_state,
    )

    final_pipeline = HeartDiseaseManualPipeline(model=final_rf)
    final_pipeline.fit(X_train, y_train)

    # Đánh giá train/test
    y_train_pred = final_pipeline.predict(X_train)
    y_test_pred = final_pipeline.predict(X_test)

    report = {
        "search_type": "manual_grid_search",
        "cv": args.cv,
        "fast_grid": args.fast,
        "selection_rule": "0.35 * recall + 0.35 * accuracy + 0.15 * precision + 0.15 * f1",
        "best_params": best["params"],
        "best_cv_metrics": best["cv_metrics"],
        "train": classification_metrics(y_train, y_train_pred),
        "test": classification_metrics(y_test, y_test_pred),
        "all_results": all_results,
        "features": final_pipeline.preprocessor.get_feature_names_out(),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Lưu pipeline gồm preprocessing + model
    final_pipeline.save(MODEL_PATH)

    # Lưu report Grid Search
    grid_report_path = REPORT_PATH.parent / "grid_search_report.json"

    with open(grid_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Ghi thêm metrics.json để project/Django vẫn đọc file quen thuộc
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Vẽ và lưu biểu đồ 4 metric trên tập test
    test_metrics_plot_path = REPORT_PATH.parent / "test_metrics_plot.png"
    plot_test_metrics(report["test"], test_metrics_plot_path)

    # In kết quả đẹp
    print_final_report(
        best=best,
        report=report,
        model_path=MODEL_PATH,
        report_path=grid_report_path,
        test_plot_path=test_metrics_plot_path,
    )


if __name__ == "__main__":
    main()
