import numpy as np


class DecisionTreeClassifierManual:
    """
    Decision Tree Classifier tự cài đặt bằng numpy.

    Mục đích:
    - Dùng để phân loại nhị phân: target = 0 hoặc target = 1
    - Không dùng scikit-learn
    - Dùng Gini impurity để chọn cách chia tốt nhất
    - Có hỗ trợ sample_weight để dùng trong Random Forest hoặc xử lý mất cân bằng lớp

    Các tham số chính:
    - max_depth: độ sâu tối đa của cây
    - min_samples_split: số mẫu tối thiểu để tiếp tục chia node
    - min_samples_leaf: số mẫu tối thiểu ở mỗi node lá
    - max_features: số feature được xét tại mỗi lần chia
    - n_thresholds: số ngưỡng thử khi tìm điểm chia
    - random_state: cố định random để kết quả lặp lại được
    """

    def __init__(
        self,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        n_thresholds=16,
        random_state=None,
    ):

        # Độ sâu tối đa của cây
        self.max_depth = max_depth

        # Nếu số mẫu trong node nhỏ hơn giá trị này thì không chia nữa
        self.min_samples_split = min_samples_split

        # Mỗi nhánh sau khi chia phải có ít nhất số mẫu này
        self.min_samples_leaf = min_samples_leaf

        # Số lượng feature được chọn ngẫu nhiên để xét tại mỗi node
        # "sqrt" nghĩa là lấy căn bậc hai của tổng số feature
        self.max_features = max_features

        # Số lượng threshold/ngưỡng tối đa được thử cho mỗi feature
        self.n_thresholds = n_thresholds

        # Random state giúp kết quả ổn định qua các lần chạy
        self.random_state = random_state

        # Bộ sinh số ngẫu nhiên của numpy
        self.rng = np.random.default_rng(random_state)

        # Node gốc của cây, ban đầu chưa có
        self.root = None

        # Số lượng feature của dữ liệu sau khi fit
        self.n_features_ = None

    def fit(self, X, y, sample_weight=None):
        """
        Huấn luyện cây quyết định.

        Tham số:
        - X: ma trận đặc trưng, shape = (số mẫu, số feature)
        - y: nhãn, gồm 0 và 1
        - sample_weight: trọng số của từng mẫu, nếu không có thì tất cả bằng 1
        """

        # Chuyển X thành numpy array kiểu float
        X = np.asarray(X, dtype=float)

        # Chuyển y thành numpy array kiểu int
        y = np.asarray(y, dtype=int)

        # Nếu không truyền sample_weight, mặc định mỗi mẫu có trọng số bằng 1
        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)

        # Lưu số lượng feature
        self.n_features_ = X.shape[1]

        # Bắt đầu xây cây từ node gốc, depth = 0
        self.root = self._build_tree(X, y, sample_weight, depth=0)

        return self

    def _gini(self, y, w):
        """
        Tính Gini impurity cho một node.

        Công thức:
        Gini = 1 - p0^2 - p1^2

        Trong đó:
        - p0 là tỷ lệ mẫu thuộc class 0
        - p1 là tỷ lệ mẫu thuộc class 1

        Gini càng nhỏ thì node càng thuần.
        """

        # Nếu node không có mẫu nào thì Gini = 0
        if len(y) == 0:
            return 0.0

        # Tổng trọng số của các mẫu trong node
        total_w = float(np.sum(w))

        # Nếu tổng trọng số không hợp lệ thì trả về 0
        if total_w <= 0:
            return 0.0

        # Tính tỷ lệ class 1 theo trọng số
        p1 = float(np.sum(w[y == 1]) / total_w)

        # Vì bài toán nhị phân nên p0 = 1 - p1
        p0 = 1.0 - p1

        # Công thức Gini impurity
        return 1.0 - p0**2 - p1**2

    def _leaf(self, y, w):
        """
        Tạo node lá.

        Node lá là node không chia tiếp nữa.
        Nó lưu:
        - prediction: nhãn dự đoán cuối cùng
        - proba: xác suất thuộc class 1
        - n_samples: số mẫu trong node
        - weighted_samples: tổng trọng số của các mẫu
        """

        # Nếu không có mẫu hoặc tổng trọng số không hợp lệ
        if len(y) == 0 or np.sum(w) <= 0:
            p1 = 0.0
        else:
            # Xác suất class 1 = tổng trọng số class 1 / tổng trọng số
            p1 = float(np.sum(w[y == 1]) / np.sum(w))

        # Nếu xác suất class 1 >= 0.5 thì dự đoán là 1, ngược lại là 0
        pred = 1 if p1 >= 0.5 else 0

        # Trả về node lá dưới dạng dictionary
        return {
            "type": "leaf",
            "prediction": pred,
            "proba": p1,
            "n_samples": int(len(y)),
            "weighted_samples": float(np.sum(w)),
        }

    def _num_features_to_try(self):
        """
        Xác định số lượng feature sẽ được xét tại mỗi node.

        Đây là điểm giống Random Forest:
        - Thay vì xét tất cả feature, mỗi node chỉ xét ngẫu nhiên một phần feature.
        """

        # Nếu max_features = "sqrt", lấy căn bậc hai của tổng số feature
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(self.n_features_)))

        # Nếu max_features = "log2", lấy log2 của tổng số feature
        if self.max_features == "log2":
            return max(1, int(np.log2(self.n_features_)))

        # Nếu max_features = None, xét tất cả feature
        if self.max_features is None:
            return self.n_features_

        # Nếu max_features là số nguyên, lấy đúng số feature đó
        if isinstance(self.max_features, int):
            return max(1, min(self.max_features, self.n_features_))

        # Nếu max_features là số thực, hiểu là tỷ lệ feature cần lấy
        # Ví dụ 0.5 nghĩa là lấy 50% số feature
        if isinstance(self.max_features, float):
            return max(
                1, min(int(self.max_features * self.n_features_), self.n_features_)
            )

        # Trường hợp còn lại, mặc định xét tất cả feature
        return self.n_features_

    def _candidate_thresholds(self, values):
        """
        Tạo danh sách các threshold/ngưỡng có thể dùng để chia dữ liệu.

        Ví dụ:
        Nếu feature age có các giá trị [40, 50, 60],
        các threshold có thể là [45, 55].
        """

        # Lấy các giá trị duy nhất của feature
        unique = np.unique(values)

        # Nếu feature chỉ có 1 giá trị duy nhất thì không thể chia
        if len(unique) <= 1:
            return []

        # Nếu số giá trị duy nhất không quá nhiều,
        # lấy trung điểm giữa các giá trị liên tiếp làm threshold
        if len(unique) <= self.n_thresholds:
            return ((unique[:-1] + unique[1:]) / 2.0).tolist()

        # Nếu có quá nhiều giá trị khác nhau,
        # lấy một số threshold theo quantile để giảm thời gian tính toán
        qs = np.linspace(0.05, 0.95, self.n_thresholds)
        thresholds = np.quantile(values, qs)

        # Loại bỏ threshold trùng nhau
        return np.unique(thresholds).tolist()

    def _best_split(self, X, y, w):
        """
        Tìm cách chia tốt nhất cho một node.

        Ý tưởng:
        - Tính Gini của node cha
        - Thử nhiều feature và threshold
        - Chọn cách chia làm giảm Gini nhiều nhất

        Gain = Gini cha - Gini trung bình có trọng số của hai node con
        """

        n_samples, n_features = X.shape

        # Tổng trọng số của node hiện tại
        total_w = float(np.sum(w))

        # Gini của node cha
        parent_gini = self._gini(y, w)

        # Nếu node đã thuần hoặc trọng số không hợp lệ thì không chia nữa
        if parent_gini == 0 or total_w <= 0:
            return None, None, 0.0

        # Xác định số feature sẽ thử tại node này
        n_try = self._num_features_to_try()

        # Chọn ngẫu nhiên một số feature để thử
        feature_indices = self.rng.choice(n_features, size=n_try, replace=False)

        # Lưu thông tin phép chia tốt nhất
        best_feature = None
        best_threshold = None
        best_gain = 0.0

        # Duyệt qua từng feature được chọn
        for feature in feature_indices:
            values = X[:, feature]

            # Duyệt qua từng threshold ứng viên của feature đó
            for threshold in self._candidate_thresholds(values):

                # Nhánh trái gồm các mẫu có giá trị feature <= threshold
                left_mask = values <= threshold

                # Nhánh phải gồm các mẫu còn lại
                right_mask = ~left_mask

                # Số mẫu ở nhánh trái và phải
                n_left = int(left_mask.sum())
                n_right = n_samples - n_left

                # Nếu một trong hai nhánh có quá ít mẫu thì bỏ qua
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                # Lấy trọng số của hai nhánh
                w_left = w[left_mask]
                w_right = w[right_mask]

                # Tổng trọng số của hai nhánh
                wl = float(np.sum(w_left))
                wr = float(np.sum(w_right))

                # Nếu trọng số không hợp lệ thì bỏ qua
                if wl <= 0 or wr <= 0:
                    continue

                # Tính Gini của nhánh trái và nhánh phải
                g_left = self._gini(y[left_mask], w_left)
                g_right = self._gini(y[right_mask], w_right)

                # Tính Gini trung bình có trọng số sau khi chia
                weighted_gini = (wl / total_w) * g_left + (wr / total_w) * g_right

                # Gain càng lớn thì phép chia càng tốt
                gain = parent_gini - weighted_gini

                # Nếu phép chia hiện tại tốt hơn phép chia tốt nhất trước đó
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = float(threshold)

        # Trả về feature, threshold và gain tốt nhất
        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, w, depth):
        """
        Xây cây quyết định bằng đệ quy.

        Tại mỗi node:
        - Nếu đạt điều kiện dừng thì tạo node lá
        - Nếu chưa dừng thì tìm cách chia tốt nhất
        - Sau đó xây tiếp cây con trái và cây con phải
        """

        # Kiểm tra cây đã đạt độ sâu tối đa chưa
        max_depth_reached = self.max_depth is not None and depth >= self.max_depth

        # Điều kiện dừng:
        # 1. Đạt độ sâu tối đa
        # 2. Số mẫu quá ít
        # 3. Node đã thuần, tức là chỉ còn một class
        if (
            max_depth_reached
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return self._leaf(y, w)

        # Tìm feature và threshold tốt nhất để chia
        feature, threshold, gain = self._best_split(X, y, w)

        # Nếu không tìm được cách chia tốt thì tạo node lá
        if feature is None or gain <= 0:
            return self._leaf(y, w)

        # Chia dữ liệu thành nhánh trái và nhánh phải
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # Trả về node hiện tại, gồm:
        # - feature dùng để chia
        # - threshold dùng để chia
        # - cây con trái
        # - cây con phải
        return {
            "type": "node",
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(
                X[left_mask], y[left_mask], w[left_mask], depth + 1
            ),
            "right": self._build_tree(
                X[right_mask], y[right_mask], w[right_mask], depth + 1
            ),
        }

    def _predict_one_proba(self, x, node):
        """
        Dự đoán xác suất class 1 cho một mẫu dữ liệu.

        Cách hoạt động:
        - Bắt đầu từ node gốc
        - Nếu x[feature] <= threshold thì đi sang trái
        - Ngược lại đi sang phải
        - Khi gặp node lá thì trả về xác suất class 1
        """

        # Lặp cho đến khi gặp node lá
        while node["type"] != "leaf":

            # Nếu giá trị feature của mẫu nhỏ hơn hoặc bằng threshold thì đi trái
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]

            # Ngược lại đi phải
            else:
                node = node["right"]

        # Trả về xác suất class 1 ở node lá
        return node["proba"]

    def predict_proba(self, X):
        """
        Dự đoán xác suất cho nhiều mẫu.

        Kết quả trả về có dạng:
        [
            [P(class 0), P(class 1)],
            [P(class 0), P(class 1)],
            ...
        ]
        """

        # Chuyển X thành numpy array
        X = np.asarray(X, dtype=float)

        # Dự đoán xác suất class 1 cho từng dòng dữ liệu
        p1 = np.array([self._predict_one_proba(row, self.root) for row in X])

        # Xác suất class 0 = 1 - xác suất class 1
        return np.vstack([1.0 - p1, p1]).T

    def predict(self, X):
        """
        Dự đoán nhãn cuối cùng cho nhiều mẫu.

        Nếu xác suất class 1 >= 0.5 thì dự đoán là 1.
        Ngược lại dự đoán là 0.
        """

        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
