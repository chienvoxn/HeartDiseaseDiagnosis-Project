import numpy as np
import pandas as pd

# Import danh sách các cột số, cột phân loại và các cột có giá trị 0 bất thường
from .config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, ZERO_AS_MISSING


class ManualPreprocessingPipeline:
    """
    Pipeline tiền xử lý dữ liệu tự code, không dùng sklearn.

    Pipeline này xử lý 2 nhóm dữ liệu:

    1. Numeric features - biến số:
        - Chuyển giá trị 0 bất thường thành NaN
        - Điền missing value bằng median
        - Chuẩn hóa dữ liệu bằng Standard Scaling

    2. Categorical features - biến phân loại:
        - Điền missing value bằng mode
        - Encode dữ liệu chữ thành số bằng One-Hot Encoding

    Lưu ý:
        - fit() chỉ dùng cho tập train
        - transform() dùng cho tập test hoặc dữ liệu người dùng nhập vào web
    """

    def __init__(self, numeric_features=None, categorical_features=None):
        """
        Hàm khởi tạo pipeline.

        Nếu không truyền danh sách cột từ bên ngoài,
        pipeline sẽ lấy mặc định từ file config.py.
        """

        # Danh sách các cột dạng số
        self.numeric_features = numeric_features or NUMERIC_FEATURES

        # Danh sách các cột dạng phân loại cần encode
        self.categorical_features = categorical_features or CATEGORICAL_FEATURES

        # Lưu median của từng cột số để xử lý missing value
        self.numeric_medians = {}

        # Lưu mean của từng cột số để chuẩn hóa
        self.numeric_means = {}

        # Lưu standard deviation của từng cột số để chuẩn hóa
        self.numeric_stds = {}

        # Lưu mode của từng cột phân loại để xử lý missing value
        self.categorical_modes = {}

        # Lưu danh sách category của từng cột để one-hot encoding
        # Ví dụ: sex -> ["Female", "Male"]
        self.categorical_categories = {}

        # Lưu tên các feature sau khi preprocessing
        self.feature_names_ = []

        # Đánh dấu pipeline đã được fit hay chưa
        self.is_fitted = False

    def _prepare_dataframe(self, X):
        """
        Chuẩn hóa dữ liệu đầu vào về dạng DataFrame.

        Hàm này giúp pipeline nhận được nhiều kiểu input:
        - dict: dữ liệu 1 bệnh nhân
        - list: danh sách nhiều bệnh nhân
        - DataFrame: dữ liệu có sẵn
        """

        # Nếu input là dict, chuyển thành DataFrame 1 dòng
        if isinstance(X, dict):
            X = pd.DataFrame([X])

        # Nếu input là list, chuyển thành DataFrame
        elif isinstance(X, list):
            X = pd.DataFrame(X)

        # Nếu input đã là DataFrame thì copy để tránh sửa dữ liệu gốc
        else:
            X = X.copy()

        # Đảm bảo dữ liệu có đủ tất cả các cột cần thiết
        # Nếu thiếu cột nào thì thêm cột đó với giá trị NaN
        for col in self.numeric_features + self.categorical_features:
            if col not in X.columns:
                X[col] = np.nan

        # Sắp xếp lại cột theo đúng thứ tự pipeline
        X = X[self.numeric_features + self.categorical_features].copy()

        # Một số cột trong dữ liệu bệnh tim không hợp lý nếu bằng 0
        # Ví dụ: huyết áp, cholesterol, nhịp tim...
        # Vì vậy đổi 0 thành NaN để lát nữa xử lý như missing value
        for col in ZERO_AS_MISSING:
            if col in X.columns:
                X[col] = X[col].replace(0, np.nan)

        return X

    def fit(self, X):
        """
        Học các thông tin cần thiết từ tập train.

        Hàm này KHÔNG trả ra dữ liệu đã xử lý.
        Nó chỉ học và lưu lại:
        - median, mean, std của cột số
        - mode của cột phân loại
        - danh sách category của từng cột phân loại
        """

        # Chuẩn hóa input về DataFrame
        X = self._prepare_dataframe(X)

        # =========================
        # 1. FIT NUMERIC FEATURES
        # =========================
        for col in self.numeric_features:
            # Ép dữ liệu sang dạng số
            # Nếu giá trị lỗi hoặc không chuyển được thì thành NaN
            values = pd.to_numeric(X[col], errors="coerce")

            # Tính median để dùng cho việc điền missing value
            median = values.median()

            # Điền missing value bằng median
            filled = values.fillna(median)

            # Tính mean sau khi đã điền missing value
            mean = filled.mean()

            # Tính độ lệch chuẩn để chuẩn hóa dữ liệu
            std = filled.std(ddof=0)

            # Nếu std bị NaN hoặc bằng 0 thì gán bằng 1
            # để tránh lỗi chia cho 0 khi scaling
            if pd.isna(std) or std == 0:
                std = 1.0

            # Lưu median của cột hiện tại
            self.numeric_medians[col] = float(median) if not pd.isna(median) else 0.0

            # Lưu mean của cột hiện tại
            self.numeric_means[col] = float(mean) if not pd.isna(mean) else 0.0

            # Lưu standard deviation của cột hiện tại
            self.numeric_stds[col] = float(std)

        # =============================
        # 2. FIT CATEGORICAL FEATURES
        # =============================
        for col in self.categorical_features:
            # Ép cột phân loại về dạng object để xử lý dữ liệu chữ
            series = X[col].astype("object")

            # Tìm mode, tức là giá trị xuất hiện nhiều nhất
            # Dùng để điền missing value cho biến phân loại
            mode_values = series.dropna().mode()

            # Nếu có mode thì lấy mode đầu tiên
            # Nếu cột toàn missing thì dùng chuỗi "missing"
            mode = mode_values.iloc[0] if len(mode_values) > 0 else "missing"

            # Điền missing value bằng mode
            filled = series.fillna(mode).astype(str)

            # Lấy danh sách tất cả category có trong cột này
            # Đây là bước "học category" để lát nữa one-hot encoding
            # Ví dụ:
            # sex có Male, Female
            # => cats = ["Female", "Male"]
            cats = sorted(filled.unique().tolist())

            # Lưu mode của cột phân loại
            self.categorical_modes[col] = mode

            # Lưu danh sách category của cột phân loại
            # Đây là thông tin quan trọng để encode ở transform()
            self.categorical_categories[col] = cats

        # ==========================================
        # 3. TẠO TÊN FEATURE SAU KHI PREPROCESSING
        # ==========================================

        self.feature_names_ = []

        # Tên các cột số sau khi scaling
        # Ví dụ: age -> num__age
        self.feature_names_.extend([f"num__{c}" for c in self.numeric_features])

        # Tên các cột sau khi one-hot encoding
        # Ví dụ:
        # sex = Female, Male
        # => cat__sex_Female, cat__sex_Male
        for col in self.categorical_features:
            for cat in self.categorical_categories[col]:
                self.feature_names_.append(f"cat__{col}_{cat}")

        # Đánh dấu pipeline đã fit xong
        self.is_fitted = True

        return self

    def transform(self, X):
        """
        Biến đổi dữ liệu dựa trên thông tin đã học từ fit().

        Hàm này thực hiện:
        - Điền missing value cho cột số
        - Standard scaling cho cột số
        - Điền missing value cho cột phân loại
        - One-hot encoding cho cột phân loại
        """

        # Không được transform nếu chưa fit
        if not self.is_fitted:
            raise RuntimeError(
                "Pipeline chưa fit. Hãy gọi fit() trên training set trước."
            )

        # Chuẩn hóa input về DataFrame
        X = self._prepare_dataframe(X)

        # Danh sách chứa các khối dữ liệu sau xử lý
        blocks = []

        # ==============================
        # 1. TRANSFORM NUMERIC FEATURES
        # ==============================

        numeric_block = []

        for col in self.numeric_features:
            # Ép dữ liệu sang số
            values = pd.to_numeric(X[col], errors="coerce")

            # Điền missing value bằng median đã học từ tập train
            values = values.fillna(self.numeric_medians[col])

            # Chuẩn hóa dữ liệu theo công thức:
            # z = (x - mean) / std
            scaled = (values - self.numeric_means[col]) / self.numeric_stds[col]

            # Chuyển cột đã xử lý thành numpy array
            numeric_block.append(scaled.to_numpy(dtype=float))

        # Ghép các cột số lại thành ma trận 2 chiều
        if numeric_block:
            blocks.append(np.vstack(numeric_block).T)

        # ==================================
        # 2. TRANSFORM CATEGORICAL FEATURES
        # ==================================
        # Đây là phần ENCODE dữ liệu phân loại bằng One-Hot Encoding

        cat_arrays = []

        for col in self.categorical_features:
            # Điền missing value bằng mode đã học từ tập train
            series = (
                X[col].astype("object").fillna(self.categorical_modes[col]).astype(str)
            )

            # Lấy danh sách category đã học ở fit()
            # Ví dụ: sex -> ["Female", "Male"]
            cats = self.categorical_categories[col]

            # Tạo ma trận toàn số 0 để chuẩn bị one-hot encoding
            # Số dòng = số mẫu dữ liệu
            # Số cột = số category của cột hiện tại
            #
            # Ví dụ có 3 bệnh nhân, sex có 2 loại Female/Male:
            # one_hot ban đầu:
            # [[0, 0],
            #  [0, 0],
            #  [0, 0]]
            one_hot = np.zeros((len(series), len(cats)), dtype=float)

            # Tạo dictionary ánh xạ category sang vị trí cột
            # Ví dụ:
            # {
            #     "Female": 0,
            #     "Male": 1
            # }
            cat_to_idx = {cat: i for i, cat in enumerate(cats)}

            # Duyệt từng dòng dữ liệu để encode
            for row_idx, value in enumerate(series):

                # Nếu giá trị có trong danh sách category đã học
                if value in cat_to_idx:
                    # Gán 1 vào đúng vị trí category đó
                    #
                    # Ví dụ:
                    # value = "Male"
                    # cat_to_idx["Male"] = 1
                    # => one_hot[row_idx, 1] = 1
                    #
                    # Kết quả:
                    # Female  Male
                    #   0       1
                    one_hot[row_idx, cat_to_idx[value]] = 1.0

                # Nếu gặp category mới chưa từng xuất hiện lúc train
                # thì giữ nguyên toàn bộ là 0
                # Cách này giống handle_unknown='ignore' trong sklearn

            # Lưu ma trận one-hot của cột hiện tại
            cat_arrays.append(one_hot)

        # Ghép one-hot của tất cả các cột phân loại lại với nhau
        if cat_arrays:
            blocks.append(np.hstack(cat_arrays))

        # Ghép numeric block và categorical block thành dữ liệu cuối cùng
        # Đây là dữ liệu số hoàn chỉnh để đưa vào Random Forest
        return np.hstack(blocks).astype(float)

    def fit_transform(self, X):
        """
        Vừa fit vừa transform dữ liệu.

        Thường dùng cho tập train.

        Tương đương:
            self.fit(X)
            self.transform(X)
        """

        return self.fit(X).transform(X)

    def get_feature_names_out(self):
        """
        Trả về danh sách tên feature sau khi preprocessing.

        Hàm này hữu ích khi:
        - Xem feature sau one-hot encoding
        - Dùng cho SHAP feature importance
        - Debug pipeline
        """

        return list(self.feature_names_)
