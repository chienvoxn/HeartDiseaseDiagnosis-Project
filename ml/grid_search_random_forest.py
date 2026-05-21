"""
Manual Grid Search

File này dùng để:
- Thử nhiều bộ hyperparameter khác nhau cho Random Forest.
- Dùng Stratified K-Fold Cross Validation để đánh giá từng bộ tham số.
- Chọn bộ tham số tốt nhất dựa trên điểm tổng hợp của 4 metric:
  Accuracy, Precision, Recall, F1-score.
- Train lại model cuối cùng bằng bộ tham số tốt nhất.
- Đánh giá model trên tập test.
- Lưu model để Django sử dụng.
"""

import argparse  # Dùng để nhận tham số khi chạy file bằng command line
import itertools  # Dùng để tạo tất cả tổ hợp hyperparameter
import json  # Dùng để lưu report dạng JSON
import time  # Dùng để đo thời gian chạy từng lần thử

import matplotlib.pyplot as plt
import numpy as np

# Import các đường dẫn cấu hình
from .config import DATA_PATH, MODEL_PATH, REPORT_PATH

# Import hàm load data và chia train/test theo tỷ lệ class
from .data_utils import load_heart_data, stratified_train_test_split

# Import pipeline gồm preprocessing + model
from .pipeline import HeartDiseaseManualPipeline

# Import Random Forest tự code
# Bên trong RandomForestClassifierManual sẽ dùng nhiều Decision Tree tự code
from .random_forest import RandomForestClassifierManual

# Import hàm tính metric: accuracy, precision, recall, f1, confusion matrix
from .metrics import classification_metrics


def stratified_kfold_indices(y, k=5, random_state=42):
    """
    Tạo Stratified K-Fold thủ công.

    Stratified K-Fold nghĩa là:
    - Chia dữ liệu thành k fold.
    - Mỗi fold vẫn giữ tỷ lệ class 0 và class 1 gần giống dữ liệu gốc.
    - Cách này phù hợp với bài toán phân loại bệnh tim.

    Ví dụ:
    Nếu dữ liệu có 45% không bệnh và 55% có bệnh,
    thì mỗi fold cũng cố gắng giữ tỷ lệ gần như vậy.
    """

    # Tạo bộ sinh số ngẫu nhiên để shuffle dữ liệu
    rng = np.random.default_rng(random_state)

    # Chuyển y thành numpy array
    y = np.asarray(y)

    # Tạo danh sách rỗng cho k fold
    folds = [[] for _ in range(k)]

    # Duyệt qua từng class, ví dụ class 0 và class 1
    for label in np.unique(y):

        # Lấy index của các mẫu thuộc class hiện tại
        idx = np.where(y == label)[0]

        # Xáo trộn index để chia ngẫu nhiên
        rng.shuffle(idx)

        # Chia index của class này thành k phần
        split_idx = np.array_split(idx, k)

        # Phân phối từng phần vào từng fold
        for fold_id, part in enumerate(split_idx):
            folds[fold_id].extend(part.tolist())

    # Tất cả index của dữ liệu
    all_indices = np.arange(len(y))

    # Danh sách kết quả, mỗi phần gồm train_idx và val_idx
    result = []

    # Tạo train/validation index cho từng fold
    for fold_id in range(k):

        # Fold hiện tại dùng làm validation
        val_idx = np.array(folds[fold_id], dtype=int)

        # Các fold còn lại dùng làm training
        train_idx = np.setdiff1d(all_indices, val_idx)

        # Shuffle lại train và validation index
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)

        # Lưu cặp train/validation index
        result.append((train_idx, val_idx))

    return result


def build_param_grid(fast=False):
    """
    Tạo danh sách các bộ hyperparameter cần thử.

    fast=True:
    - Dùng grid nhỏ để test nhanh.
    - Phù hợp khi muốn kiểm tra code có chạy đúng không.

    fast=False:
    - Dùng grid lớn hơn để fine-tune chính thức.
    - Sẽ chạy lâu hơn nhưng tìm được bộ tham số tốt hơn.

    Các hyperparameter chính:
    - n_estimators: số cây trong Random Forest
    - max_depth: độ sâu tối đa của mỗi cây
    - min_samples_split: số mẫu tối thiểu để chia node
    - min_samples_leaf: số mẫu tối thiểu ở node lá
    - max_features: số feature được xét tại mỗi lần chia
    - class_weight: xử lý mất cân bằng class
    - n_thresholds: số ngưỡng thử khi chia node
    """

    if fast:
        # Grid nhỏ để chạy nhanh
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
        # Grid lớn hơn để fine-tuning chính thức
        grid = {
            "n_estimators": [120, 200],
            "max_depth": [4, 5, 6, 7],
            "min_samples_split": [10, 15, 20],
            "min_samples_leaf": [4, 6, 8],
            "max_features": ["sqrt"],
            "class_weight": [None],
            "n_thresholds": [12, 20],
        }

    # Lấy danh sách tên hyperparameter
    keys = list(grid.keys())

    # itertools.product tạo tất cả tổ hợp có thể có
    # Ví dụ:
    # max_depth = [4, 5]
    # min_samples_leaf = [4, 6]
    # => tạo 4 tổ hợp
    for values in itertools.product(*[grid[k] for k in keys]):

        # Ghép tên tham số với giá trị tương ứng
        yield dict(zip(keys, values))


def evaluate_params_cv(X_train, y_train, params, cv=5, random_state=42):
    """
    Đánh giá một bộ hyperparameter bằng Stratified K-Fold Cross Validation.

    Quy trình:
    1. Chia training set thành cv fold.
    2. Với mỗi fold:
       - Train model trên cv-1 fold.
       - Validate trên fold còn lại.
    3. Tính metric cho từng fold.
    4. Lấy trung bình metric của tất cả fold.

    Điểm quan trọng:
    - Mỗi fold có pipeline riêng.
    - Preprocessing chỉ fit trên train fold.
    - Sau đó transform validation fold.
    - Cách này giúp tránh data leakage.
    """

    # Chuyển y_train thành numpy array
    y_train_arr = np.asarray(y_train)

    # Tạo danh sách các fold
    folds = stratified_kfold_indices(y_train_arr, k=cv, random_state=random_state)

    # Lưu metric của từng fold
    fold_metrics = []

    # Duyệt qua từng fold
    for fold_id, (tr_idx, val_idx) in enumerate(folds, start=1):

        # Lấy dữ liệu train của fold hiện tại
        X_tr = X_train.iloc[tr_idx]
        y_tr = y_train_arr[tr_idx]

        # Lấy dữ liệu validation của fold hiện tại
        X_val = X_train.iloc[val_idx]
        y_val = y_train_arr[val_idx]

        # Tạo Random Forest với bộ hyperparameter đang thử
        model = RandomForestClassifierManual(
            **params,
            bootstrap=True,
            random_state=random_state + fold_id,
        )

        # Tạo pipeline gồm preprocessing + Random Forest
        pipeline = HeartDiseaseManualPipeline(model=model)

        # Fit pipeline trên train fold
        pipeline.fit(X_tr, y_tr)

        # Dự đoán trên validation fold
        y_pred = pipeline.predict(X_val)

        # Tính metric trên validation fold
        metrics = classification_metrics(y_val, y_pred)

        # Lưu metric của fold này
        fold_metrics.append(metrics)

    # Tính trung bình metric của tất cả fold
    avg = {
        "accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "precision": float(np.mean([m["precision"] for m in fold_metrics])),
        "recall": float(np.mean([m["recall"] for m in fold_metrics])),
        "f1": float(np.mean([m["f1"] for m in fold_metrics])),
    }

    return avg, fold_metrics


def selection_score(metrics):
    """
    Tính điểm tổng hợp để chọn model tốt nhất.

    Vì đây là bài toán bệnh tim:
    - Recall quan trọng vì cần hạn chế bỏ sót bệnh nhân có bệnh.
    - Accuracy cũng quan trọng để đảm bảo mô hình đúng tổng thể.
    - Precision giúp giảm cảnh báo nhầm.
    - F1-score giúp cân bằng Precision và Recall.

    Công thức:
    score = 0.35 * recall
          + 0.35 * accuracy
          + 0.15 * precision
          + 0.15 * f1
    """

    return (
        0.35 * metrics["recall"]
        + 0.35 * metrics["accuracy"]
        + 0.15 * metrics["precision"]
        + 0.15 * metrics["f1"]
    )


def grid_search(X_train, y_train, cv=5, random_state=42, fast=False):
    """
    Chạy Grid Search thủ công.

    Grid Search nghĩa là:
    - Thử toàn bộ tổ hợp hyperparameter.
    - Mỗi bộ hyperparameter được đánh giá bằng Cross Validation.
    - Sau đó chọn bộ có selection_score cao nhất.
    """

    # Tạo danh sách tất cả bộ hyperparameter cần thử
    params_list = list(build_param_grid(fast=fast))

    # Tổng số tổ hợp cần thử
    total = len(params_list)

    # Lưu kết quả của tất cả lần thử
    results = []

    print(f"Start manual GridSearchCV | candidates={total} | cv={cv}")

    # Duyệt qua từng bộ hyperparameter
    for i, params in enumerate(params_list, start=1):

        # Bắt đầu tính thời gian
        start = time.time()

        # Đánh giá bộ params hiện tại bằng Cross Validation
        avg_metrics, fold_metrics = evaluate_params_cv(
            X_train,
            y_train,
            params,
            cv=cv,
            random_state=random_state + i * 10,
        )

        # Tính thời gian chạy
        elapsed = time.time() - start

        # Lưu kết quả của lần thử hiện tại
        row = {
            "iteration": i,
            "params": params,
            "cv_metrics": avg_metrics,
            "fold_metrics": fold_metrics,
            "elapsed_seconds": round(elapsed, 2),
        }

        results.append(row)

        # In thông tin để theo dõi quá trình chạy
        print(
            f"[{i}/{total}] "
            f"Recall={avg_metrics['recall']:.4f} | "
            f"F1={avg_metrics['f1']:.4f} | "
            f"Acc={avg_metrics['accuracy']:.4f} | "
            f"Params={params} | {elapsed:.1f}s"
        )

    # Sắp xếp kết quả theo selection_score giảm dần
    results_sorted = sorted(
        results,
        key=lambda r: selection_score(r["cv_metrics"]),
        reverse=True,
    )

    # Trả về:
    # - kết quả tốt nhất
    # - toàn bộ kết quả đã sắp xếp
    return results_sorted[0], results_sorted


def print_section_title(title):
    """
    In tiêu đề section cho đẹp khi chạy terminal.
    """

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_params(params):
    """
    In bộ hyperparameter tốt nhất.
    """

    print("\nBest Hyperparameters:")
    print("-" * 60)

    for key, value in params.items():
        print(f"{key:<25}: {value}")


def print_metrics(title, metrics):
    """
    In các metric gồm:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    """

    print(f"\n{title}")
    print("-" * 60)
    print(f"{'Accuracy':<15}: {metrics['accuracy']:.4f}")
    print(f"{'Precision':<15}: {metrics['precision']:.4f}")
    print(f"{'Recall':<15}: {metrics['recall']:.4f}")
    print(f"{'F1-score':<15}: {metrics['f1']:.4f}")


def print_confusion_matrix(cm):
    """
    In confusion matrix theo dạng dễ hiểu.

    Confusion Matrix:
    - TN: Thực tế không bệnh, dự đoán không bệnh
    - FP: Thực tế không bệnh, dự đoán có bệnh
    - FN: Thực tế có bệnh, dự đoán không bệnh
    - TP: Thực tế có bệnh, dự đoán có bệnh
    """

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
    """
    Vẽ và lưu biểu đồ 4 metric trên tập test sau fine-tuning.

    Biểu đồ gồm:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    """

    metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]

    metric_values = [
        test_metrics["accuracy"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
    ]

    plt.figure(figsize=(8, 5))

    # Vẽ biểu đồ cột
    bars = plt.bar(metric_names, metric_values)

    plt.ylim(0, 1)
    plt.title("Final Test Metrics After Fine-tuning")
    plt.xlabel("Metric")
    plt.ylabel("Score")

    # Ghi giá trị lên đầu từng cột
    for bar, value in zip(bars, metric_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    # Lưu biểu đồ ra file ảnh
    plt.savefig(save_path, dpi=300)

    # Đóng figure để tránh tốn bộ nhớ
    plt.close()


def print_final_report(best, report, model_path, report_path, test_plot_path=None):
    """
    In báo cáo cuối cùng sau khi Grid Search xong.
    """

    print_section_title("GRID SEARCH RESULT")

    # In bộ tham số tốt nhất
    print_params(best["params"])

    # In metric Cross Validation tốt nhất
    print_metrics("Best Cross-Validation Metrics", best["cv_metrics"])

    # In metric trên tập train
    print_metrics("Train Metrics", report["train"])

    # In metric trên tập test
    print_metrics("Final Test Metrics", report["test"])

    # In confusion matrix trên tập test
    print_confusion_matrix(report["test"]["confusion_matrix"])

    print_section_title("SAVED FILES")

    # In đường dẫn các file đã lưu
    print("Saved model to:", model_path)
    print("Saved grid search report to:", report_path)

    if test_plot_path is not None:
        print("Saved test metrics plot to:", test_plot_path)


def main():
    """
    Hàm chính của file.

    Khi chạy:
        python -m ml.grid_search_random_forest

    Chương trình sẽ:
    1. Load dữ liệu.
    2. Chia train/test.
    3. Chạy Grid Search trên training set.
    4. Chọn bộ tham số tốt nhất.
    5. Train lại final model trên toàn bộ training set.
    6. Đánh giá trên train/test.
    7. Lưu model, report và biểu đồ metric.
    """

    # Tạo parser để nhận tham số từ command line
    parser = argparse.ArgumentParser()

    # Số fold dùng trong Cross Validation
    parser.add_argument("--cv", type=int, default=5, help="Số fold cross-validation")

    # Random state để cố định kết quả
    parser.add_argument("--random-state", type=int, default=42)

    # Nếu thêm --fast khi chạy thì dùng grid nhỏ
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Dùng grid nhỏ để test nhanh",
    )

    # Đọc tham số từ command line
    args = parser.parse_args()

    # Load dữ liệu đã được xử lý cơ bản trong data_utils
    X, y = load_heart_data(DATA_PATH)

    # Chia dữ liệu thành train/test theo tỷ lệ class
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.random_state,
    )

    # Chạy Grid Search trên training set
    best, all_results = grid_search(
        X_train,
        y_train,
        cv=args.cv,
        random_state=args.random_state,
        fast=args.fast,
    )

    # Tạo final Random Forest bằng bộ tham số tốt nhất
    final_rf = RandomForestClassifierManual(
        **best["params"],
        bootstrap=True,
        random_state=args.random_state,
    )

    # Tạo pipeline cuối cùng gồm preprocessing + final Random Forest
    final_pipeline = HeartDiseaseManualPipeline(model=final_rf)

    # Train final pipeline trên toàn bộ training set
    final_pipeline.fit(X_train, y_train)

    # Dự đoán trên tập train
    y_train_pred = final_pipeline.predict(X_train)

    # Dự đoán trên tập test
    y_test_pred = final_pipeline.predict(X_test)

    # Tạo report tổng hợp
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

    # Tạo thư mục lưu model nếu chưa có
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Tạo thư mục lưu report nếu chưa có
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Lưu pipeline gồm preprocessing + model
    # File này thường là .pkl để Django load lên dự đoán
    final_pipeline.save(MODEL_PATH)

    # Đường dẫn file report Grid Search
    grid_report_path = REPORT_PATH.parent / "grid_search_report.json"

    # Lưu report Grid Search ra file JSON
    with open(grid_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Ghi thêm metrics.json để project hoặc Django vẫn đọc file quen thuộc
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Vẽ và lưu biểu đồ 4 metric trên tập test
    test_metrics_plot_path = REPORT_PATH.parent / "test_metrics_plot.png"
    plot_test_metrics(report["test"], test_metrics_plot_path)

    # In kết quả cuối cùng ra terminal
    print_final_report(
        best=best,
        report=report,
        model_path=MODEL_PATH,
        report_path=grid_report_path,
        test_plot_path=test_metrics_plot_path,
    )


# Nếu chạy trực tiếp file này thì gọi hàm main()
if __name__ == "__main__":
    main()
