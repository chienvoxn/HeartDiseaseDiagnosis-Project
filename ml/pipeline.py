# Import thư viện pickle để lưu/load toàn bộ pipeline ra file .pkl
import pickle

# Import pipeline tiền xử lý dữ liệu tự code
# Pipeline này xử lý missing value, scaling, one-hot encoding...
from .preprocessing import ManualPreprocessingPipeline

# Import mô hình Random Forest tự code
# Bên trong RandomForestClassifierManual sẽ dùng nhiều Decision Tree tự code
from .random_forest import RandomForestClassifierManual


class HeartDiseaseManualPipeline:
    """
    Pipeline đầy đủ cho bài toán dự đoán bệnh tim.

    Pipeline này gồm 2 phần chính:

    1. Preprocessing:
       - Xử lý dữ liệu đầu vào
       - Điền missing value
       - Chuẩn hóa biến số
       - One-hot encoding biến phân loại

    2. Model:
       - Random Forest tự code
       - Random Forest này dùng nhiều Decision Tree làm cây con

    Mục đích:
    - Gộp preprocessing và model vào chung một pipeline
    - Khi train, dữ liệu sẽ được xử lý trước rồi mới đưa vào model
    - Khi predict, dữ liệu người dùng nhập cũng được xử lý giống lúc train
    - Có thể lưu toàn bộ pipeline thành file .pkl để Django sử dụng
    """

    def __init__(self, preprocessor=None, model=None):
        """
        Hàm khởi tạo pipeline.

        Tham số:
        - preprocessor:
          Pipeline tiền xử lý dữ liệu.
          Nếu không truyền vào thì dùng ManualPreprocessingPipeline mặc định.

        - model:
          Mô hình Machine Learning.
          Nếu không truyền vào thì dùng RandomForestClassifierManual mặc định.
        """

        # Nếu có preprocessor truyền vào thì dùng preprocessor đó
        # Nếu không thì tạo ManualPreprocessingPipeline mới
        self.preprocessor = preprocessor or ManualPreprocessingPipeline()

        # Nếu có model truyền vào thì dùng model đó
        # Nếu không thì tạo RandomForestClassifierManual mới
        self.model = model or RandomForestClassifierManual()

    def fit(self, X, y):
        """
        Huấn luyện toàn bộ pipeline.

        Quy trình:
        1. Fit preprocessor trên tập train.
        2. Transform X thành dữ liệu số hoàn chỉnh.
        3. Fit Random Forest trên dữ liệu đã xử lý.

        Tham số:
        - X: dữ liệu đầu vào, thường là DataFrame
        - y: nhãn target, gồm 0 và 1

        Trả về:
        - self: chính pipeline sau khi train
        """

        # Fit preprocessing trên X rồi transform X
        # Kết quả X_processed là dữ liệu toàn số,
        # đã xử lý missing value, scaling và one-hot encoding
        X_processed = self.preprocessor.fit_transform(X)

        # Train model Random Forest trên dữ liệu đã xử lý
        self.model.fit(X_processed, y)

        # Trả về chính object pipeline
        return self

    def predict_proba(self, X):
        """
        Dự đoán xác suất cho dữ liệu đầu vào.

        Quy trình:
        1. Transform X bằng preprocessor đã fit từ trước.
        2. Đưa dữ liệu đã xử lý vào model.
        3. Trả về xác suất dự đoán.

        Kết quả trả về có dạng:
        [
            [P(không bệnh), P(có bệnh)],
            [P(không bệnh), P(có bệnh)],
            ...
        ]
        """

        # Transform dữ liệu đầu vào theo đúng cách đã học từ tập train
        X_processed = self.preprocessor.transform(X)

        # Dự đoán xác suất bằng Random Forest
        return self.model.predict_proba(X_processed)

    def predict(self, X):
        """
        Dự đoán nhãn cuối cùng cho dữ liệu đầu vào.

        Quy trình:
        1. Transform X bằng preprocessor.
        2. Đưa dữ liệu đã xử lý vào model.
        3. Model trả về nhãn 0 hoặc 1.

        Ý nghĩa:
        - 0: Không bệnh
        - 1: Có bệnh
        """

        # Transform dữ liệu đầu vào
        X_processed = self.preprocessor.transform(X)

        # Dự đoán nhãn bằng Random Forest
        return self.model.predict(X_processed)

    def save(self, path):
        """
        Lưu toàn bộ pipeline ra file .pkl.

        File được lưu gồm:
        - Preprocessor đã fit
        - Random Forest đã train
        - Các thông tin median, mean, std, category...
        - Các cây Decision Tree bên trong Random Forest

        Mục đích:
        - Sau khi train xong, lưu model lại.
        - Django có thể load file .pkl này để dự đoán mà không cần train lại.
        """

        # Mở file ở chế độ ghi nhị phân
        with open(path, "wb") as f:

            # Lưu toàn bộ object pipeline vào file
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        Load pipeline đã lưu từ file .pkl.

        Đây là staticmethod nên có thể gọi trực tiếp:

            model = HeartDiseaseManualPipeline.load(MODEL_PATH)

        Sau khi load:
        - Có thể dùng model.predict()
        - Có thể dùng model.predict_proba()
        - Không cần fit lại
        """

        # Mở file .pkl ở chế độ đọc nhị phân
        with open(path, "rb") as f:

            # Load lại object pipeline đã lưu
            return pickle.load(f)
