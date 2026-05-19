import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Tính Accuracy.

    Accuracy = Số dự đoán đúng / Tổng số mẫu

    Ý nghĩa:
    - Cho biết mô hình dự đoán đúng bao nhiêu phần trăm trên toàn bộ dữ liệu.
    """

    # Chuyển y_true và y_pred về numpy array để dễ tính toán
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # So sánh từng phần tử y_true và y_pred
    # Nếu bằng nhau thì True, khác nhau thì False
    # np.mean sẽ lấy trung bình số True
    return float(np.mean(y_true == y_pred))


def precision_score(y_true, y_pred):
    """
    Tính Precision.

    Precision = TP / (TP + FP)

    Trong bài toán bệnh tim:
    - TP: Dự đoán có bệnh và thực tế có bệnh
    - FP: Dự đoán có bệnh nhưng thực tế không bệnh

    Ý nghĩa:
    - Trong tất cả những người mô hình dự đoán là có bệnh,
      có bao nhiêu người thật sự có bệnh.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # True Positive: thực tế là 1, dự đoán cũng là 1
    tp = np.sum((y_true == 1) & (y_pred == 1))

    # False Positive: thực tế là 0, nhưng dự đoán là 1
    fp = np.sum((y_true == 0) & (y_pred == 1))

    # Tránh lỗi chia cho 0
    return float(tp / (tp + fp)) if (tp + fp) else 0.0


def recall_score(y_true, y_pred):
    """
    Tính Recall.

    Recall = TP / (TP + FN)

    Trong bài toán bệnh tim:
    - TP: Dự đoán có bệnh và thực tế có bệnh
    - FN: Dự đoán không bệnh nhưng thực tế có bệnh

    Ý nghĩa:
    - Trong tất cả những người thật sự có bệnh,
      mô hình phát hiện đúng được bao nhiêu người.

    Với bài toán y tế, Recall thường rất quan trọng,
    vì bỏ sót người có bệnh là nguy hiểm.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # True Positive: thực tế là 1, dự đoán là 1
    tp = np.sum((y_true == 1) & (y_pred == 1))

    # False Negative: thực tế là 1, nhưng dự đoán là 0
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Tránh lỗi chia cho 0
    return float(tp / (tp + fn)) if (tp + fn) else 0.0


def f1_score(y_true, y_pred):
    """
    Tính F1-score.

    F1 = 2 * Precision * Recall / (Precision + Recall)

    Ý nghĩa:
    - F1 là trung bình điều hòa giữa Precision và Recall.
    - Dùng khi muốn cân bằng giữa việc dự đoán đúng class 1
      và không bỏ sót class 1.
    """

    # Tính precision
    p = precision_score(y_true, y_pred)

    # Tính recall
    r = recall_score(y_true, y_pred)

    # Tránh lỗi chia cho 0
    return float(2 * p * r / (p + r)) if (p + r) else 0.0


def confusion_matrix_binary(y_true, y_pred):
    """
    Tính confusion matrix cho bài toán binary classification.

    Quy ước:
    - 0: Không bệnh
    - 1: Có bệnh

    Confusion matrix dạng:

                    Predicted 0    Predicted 1
    Actual 0            TN             FP
    Actual 1            FN             TP

    Trong đó:
    - TN: Thực tế không bệnh, dự đoán không bệnh
    - FP: Thực tế không bệnh, dự đoán có bệnh
    - FN: Thực tế có bệnh, dự đoán không bệnh
    - TP: Thực tế có bệnh, dự đoán có bệnh
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # True Negative: thực tế là 0, dự đoán là 0
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    # False Positive: thực tế là 0, dự đoán là 1
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))

    # False Negative: thực tế là 1, dự đoán là 0
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    # True Positive: thực tế là 1, dự đoán là 1
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    return [[tn, fp], [fn, tp]]


def print_confusion_matrix(y_true, y_pred):
    """
    In confusion matrix rõ ràng ra màn hình.

    Hàm này giúp dễ đọc hơn thay vì chỉ in:
    [[TN, FP], [FN, TP]]
    """

    # Lấy confusion matrix
    cm = confusion_matrix_binary(y_true, y_pred)

    # Tách từng giá trị ra cho dễ dùng
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    print("Confusion Matrix:")
    print()
    print("                 Predicted 0      Predicted 1")
    print("                 Không bệnh       Có bệnh")
    print(f"Actual 0 Không bệnh     {tn:<12}     {fp:<12}")
    print(f"Actual 1 Có bệnh        {fn:<12}     {tp:<12}")
    print()
    print("Giải thích:")
    print(f"TN = {tn}: Thực tế không bệnh, mô hình dự đoán không bệnh")
    print(f"FP = {fp}: Thực tế không bệnh, mô hình dự đoán có bệnh")
    print(f"FN = {fn}: Thực tế có bệnh, mô hình dự đoán không bệnh")
    print(f"TP = {tp}: Thực tế có bệnh, mô hình dự đoán có bệnh")


def classification_metrics(y_true, y_pred):
    """
    Trả về toàn bộ các metric đánh giá mô hình.

    Gồm:
    - accuracy
    - precision
    - recall
    - f1
    - confusion_matrix
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix_binary(y_true, y_pred),
    }


def print_classification_report(y_true, y_pred):
    """
    In toàn bộ kết quả đánh giá mô hình một cách rõ ràng.
    """

    metrics = classification_metrics(y_true, y_pred)

    print("Classification Metrics:")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print()

    print_confusion_matrix(y_true, y_pred)