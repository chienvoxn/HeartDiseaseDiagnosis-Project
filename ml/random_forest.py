import numpy as np

# Import class Decision Tree tự code ở file tree.py
# Mỗi cây trong Random Forest sẽ là một DecisionTreeClassifierManual
from .tree import DecisionTreeClassifierManual


class RandomForestClassifierManual:
    """
    Random Forest Classifier tự cài đặt bằng numpy.

    Ý tưởng chính của Random Forest:
    - Tạo nhiều cây Decision Tree khác nhau.
    - Mỗi cây được train trên một tập dữ liệu bootstrap khác nhau.
    - Bootstrap sampling nghĩa là lấy mẫu ngẫu nhiên có lặp lại từ tập train.
    - Mỗi cây sẽ học hơi khác nhau.
    - Khi dự đoán, Random Forest lấy kết quả trung bình từ nhiều cây.
    - Với bài toán classification, có thể hiểu là voting hoặc averaging probability.

    Code này dùng:
    - DecisionTreeClassifierManual làm cây con.
    - Bootstrap sampling để tạo dữ liệu train riêng cho từng cây.
    - Probability averaging để dự đoán xác suất cuối cùng.
    - Threshold 0.5 để chuyển xác suất thành nhãn 0 hoặc 1.
    """

    def __init__(
        self,
        n_estimators=120,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        n_thresholds=16,
        class_weight=None,
        bootstrap=True,
        random_state=42,
    ):
        """
        Hàm khởi tạo Random Forest.

        Các tham số:

        n_estimators:
        - Số lượng cây Decision Tree trong rừng.
        - Ví dụ n_estimators=120 nghĩa là tạo 120 cây.

        max_depth:
        - Độ sâu tối đa của mỗi cây.
        - Giúp tránh overfitting.

        min_samples_split:
        - Số mẫu tối thiểu để một node được tiếp tục chia.

        min_samples_leaf:
        - Số mẫu tối thiểu ở mỗi node lá.

        max_features:
        - Số feature được xét ngẫu nhiên tại mỗi node của Decision Tree.
        - "sqrt" nghĩa là mỗi lần chia chỉ xét sqrt(tổng số feature).
        - Đây là đặc điểm quan trọng của Random Forest.

        n_thresholds:
        - Số lượng ngưỡng thử khi tìm điểm chia cho mỗi feature.

        class_weight:
        - None: không cân bằng trọng số lớp.
        - "balanced": tự động tăng trọng số cho class ít mẫu hơn.

        bootstrap:
        - True: mỗi cây train trên một tập bootstrap khác nhau.
        - False: mỗi cây train trên toàn bộ dữ liệu.

        random_state:
        - Cố định random để kết quả chạy lại giống nhau.
        """

        # Số lượng cây trong Random Forest
        self.n_estimators = n_estimators

        # Độ sâu tối đa của từng cây
        self.max_depth = max_depth

        # Số mẫu tối thiểu để một node tiếp tục được chia
        self.min_samples_split = min_samples_split

        # Số mẫu tối thiểu ở node lá
        self.min_samples_leaf = min_samples_leaf

        # Số feature được xét tại mỗi node
        self.max_features = max_features

        # Số threshold được thử cho mỗi feature
        self.n_thresholds = n_thresholds

        # Cách xử lý mất cân bằng class
        self.class_weight = class_weight

        # Có dùng bootstrap sampling hay không
        self.bootstrap = bootstrap

        # Giá trị random để kết quả ổn định
        self.random_state = random_state

        # Danh sách lưu các cây Decision Tree sau khi train
        self.trees = []

        # Bộ sinh số ngẫu nhiên của numpy
        self.rng = np.random.default_rng(random_state)

    def _compute_sample_weight(self, y):
        """
        Tính trọng số cho từng mẫu dữ liệu.

        Mục đích:
        - Nếu dữ liệu bị lệch class, ví dụ class 1 nhiều hơn class 0,
          ta có thể tăng trọng số cho class ít mẫu hơn.
        - Điều này giúp mô hình chú ý hơn đến class thiểu số.

        Nếu class_weight=None:
        - Mỗi mẫu có trọng số bằng 1.

        Nếu class_weight="balanced":
        - Class nào ít mẫu hơn sẽ có trọng số cao hơn.
        """

        # Chuyển y về numpy array kiểu int
        y = np.asarray(y, dtype=int)

        # Mặc định tất cả mẫu có trọng số bằng 1
        weights = np.ones(len(y), dtype=float)

        # Nếu chọn balanced thì tính lại trọng số theo số lượng từng class
        if self.class_weight == "balanced":

            # classes: danh sách các class, ví dụ [0, 1]
            # counts: số lượng mẫu của từng class
            classes, counts = np.unique(y, return_counts=True)

            # Tổng số mẫu
            n_samples = len(y)

            # Số lượng class
            n_classes = len(classes)

            # Công thức giống sklearn:
            # weight = n_samples / (n_classes * số mẫu của class đó)
            weight_map = {
                cls: n_samples / (n_classes * cnt) for cls, cnt in zip(classes, counts)
            }

            # Gán trọng số cho từng mẫu dựa trên nhãn của nó
            weights = np.array([weight_map[label] for label in y], dtype=float)

        # Trả về mảng trọng số
        return weights

    def fit(self, X, y):
        """
        Huấn luyện Random Forest.

        Quy trình:
        1. Chuyển X, y về numpy array.
        2. Tính sample_weight nếu cần.
        3. Lặp n_estimators lần.
        4. Mỗi lần tạo một tập bootstrap.
        5. Tạo một Decision Tree mới.
        6. Train cây đó.
        7. Lưu cây vào self.trees.
        """

        # Chuyển X thành numpy array kiểu float
        X = np.asarray(X, dtype=float)

        # Chuyển y thành numpy array kiểu int
        y = np.asarray(y, dtype=int)

        # Số lượng mẫu trong tập train
        n_samples = X.shape[0]

        # Tính trọng số cho từng mẫu
        base_weight = self._compute_sample_weight(y)

        # Reset danh sách cây trước khi train
        self.trees = []

        # Reset random generator để kết quả ổn định khi fit lại
        self.rng = np.random.default_rng(self.random_state)

        # Tạo lần lượt từng cây trong Random Forest
        for i in range(self.n_estimators):

            # Nếu bootstrap=True:
            # Lấy ngẫu nhiên n_samples dòng từ tập train, có lặp lại
            # Ví dụ một dòng có thể xuất hiện nhiều lần, một số dòng có thể không được chọn
            if self.bootstrap:
                indices = self.rng.integers(0, n_samples, size=n_samples)

            # Nếu bootstrap=False:
            # Dùng toàn bộ dữ liệu train cho mỗi cây
            else:
                indices = np.arange(n_samples)

            # Tạo một cây Decision Tree mới
            tree = DecisionTreeClassifierManual(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_thresholds=self.n_thresholds,
                # Mỗi cây có random_state khác nhau để tạo sự đa dạng
                random_state=(
                    self.random_state + i if self.random_state is not None else None
                ),
            )

            # Train cây trên tập bootstrap
            # X[indices]: dữ liệu train con
            # y[indices]: nhãn tương ứng
            # base_weight[indices]: trọng số tương ứng với các mẫu được chọn
            tree.fit(X[indices], y[indices], sample_weight=base_weight[indices])

            # Lưu cây đã train vào danh sách trees
            self.trees.append(tree)

        # Trả về chính mô hình sau khi train
        return self

    def predict_proba(self, X):
        """
        Dự đoán xác suất cho từng mẫu.

        Cách hoạt động:
        - Mỗi cây Decision Tree dự đoán xác suất class 1.
        - Random Forest lấy trung bình xác suất class 1 của tất cả cây.
        - Xác suất class 0 = 1 - xác suất class 1.

        Kết quả trả về có dạng:
        [
            [P(class 0), P(class 1)],
            [P(class 0), P(class 1)],
            ...
        ]
        """

        # Nếu chưa có cây nào, tức là model chưa được train
        if not self.trees:
            raise RuntimeError("Model chưa được train.")

        # Lấy xác suất class 1 từ từng cây
        # Mỗi tree.predict_proba(X) trả về 2 cột: P(0), P(1)
        # Ta lấy cột [:, 1], tức là xác suất class 1
        probs = np.array([tree.predict_proba(X)[:, 1] for tree in self.trees])

        # Lấy trung bình xác suất class 1 của tất cả cây
        p1 = probs.mean(axis=0)

        # Ghép xác suất class 0 và class 1 lại
        # P(class 0) = 1 - p1
        # P(class 1) = p1
        return np.vstack([1.0 - p1, p1]).T

    def predict(self, X):
        """
        Dự đoán nhãn cuối cùng cho từng mẫu.

        Nếu P(class 1) >= 0.5:
        - Dự đoán là 1, tức là có bệnh.

        Nếu P(class 1) < 0.5:
        - Dự đoán là 0, tức là không bệnh.
        """

        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
