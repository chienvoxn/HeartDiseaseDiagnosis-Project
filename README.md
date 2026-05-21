# Heart Disease Diagnosis - Manual Random Forest + Django

Project dự đoán nguy cơ bệnh tim dựa trên bộ dữ liệu **Heart Disease UCI**.  
Điểm chính của project là xây dựng pipeline Machine Learning rõ ràng, tự code các thành phần cốt lõi và triển khai mô hình lên ứng dụng web bằng Django.

---

## Mục tiêu project

- Xây dựng pipeline Machine Learning rõ ràng, dễ hiểu và dễ demo.
- Không dùng `scikit-learn` cho phần train model chính.
- Xây dựng **Decision Tree** và **Random Forest** thủ công.
- Fine-tuning mô hình bằng **Manual Grid Search**.
- Lưu mô hình bằng `pickle`.
- Deploy local bằng **Django**.
- Xây dựng giao diện dashboard trực quan cho dự đoán và đánh giá kết quả.
- Kết hợp giải thích kết quả dự đoán bằng **rule-based logic**.

---

## Công nghệ sử dụng

### Machine Learning

| Công nghệ | Vai trò |
|---|---|
| **Python** | Ngôn ngữ lập trình chính |
| **NumPy** | Tính toán số học, xử lý mảng |
| **Pandas** | Đọc và xử lý dữ liệu dạng bảng |
| **Manual Decision Tree** | Cây quyết định tự cài đặt |
| **Manual Random Forest** | Mô hình chính của project |
| **Manual Grid Search** | Fine-tuning siêu tham số |
| **Pickle** | Lưu và tải lại model |

### Web Deployment

| Công nghệ | Vai trò |
|---|---|
| **Django** | Framework xây dựng web app |
| **HTML/CSS** | Xây dựng giao diện người dùng |
| **Django Template Engine** | Render dữ liệu ra giao diện |

---

## Lưu ý về thư viện

Các thành phần sau được **tự code**, không dùng `scikit-learn` trong phần model chính:

- Preprocessing pipeline
- Train/test split
- Metrics
- Decision Tree
- Random Forest
- Grid Search
- Prediction pipeline

> Project tập trung vào việc hiểu rõ cách hoạt động của Random Forest thay vì chỉ sử dụng thư viện có sẵn.

---

## Cấu trúc project

```text
HeartDiseaseDiagnosis-Project/
│
├── data/
│   └── heart_disease_uci.csv
│
├── notebooks/
│   └── HeartDiseaseDiagnosis.ipynb
│
├── ml/
│   ├── models/
│   │   └── manual_random_forest.pkl
│   ├── reports/
│   ├── config.py
│   ├── data_utils.py
│   ├── preprocessing.py
│   ├── tree.py
│   ├── random_forest.py
│   ├── metrics.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── train.py
│   └── grid_search_random_forest.py
│
├── web/
│   ├── manage.py
│   ├── db.sqlite3
│   ├── heart_project/
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── media/
│   └── diagnosis/
│       ├── static/
│       │   └── diagnosis/
│       │       └── style.css
│       ├── templates/
│       │   └── diagnosis/
│       │       ├── base_dashboard.html
│       │       ├── dashboard.html
│       │       ├── evaluation.html
│       │       ├── index.html
│       │       ├── login.html
│       │       └── predict.html
│       ├── templatetags/
│       │   ├── __init__.py
│       │   └── custom_filters.py
│       ├── forms.py
│       ├── views.py
│       ├── urls.py
│       └── apps.py
│
├── requirements.txt
└── README.md
```

---

## Pipeline Machine Learning

```text
Raw data
→ tạo target binary từ cột num
→ loại bỏ id, dataset, num
→ xử lý giá trị thiếu
→ custom preprocessing:
   - numeric:
       + median imputation
       + standard scaling
   - categorical:
       + mode imputation
       + one-hot encoding
→ stratified train/test split
→ Manual Random Forest
→ evaluate:
   - Accuracy
   - Precision
   - Recall
   - F1-score
→ Manual Grid Search để fine-tune
→ lưu model bằng pickle
→ Django load model để predict
```

---

## Chức năng hệ thống

### Trang chủ

- Giới thiệu hệ thống dự đoán bệnh tim.
- Hiển thị thông tin tổng quan về mô hình và dữ liệu.

### Dashboard

- Tổng quan dataset.
- Số lượng bệnh nhân.
- Thống kê nguy cơ bệnh tim.
- Hiển thị thông tin đánh giá mô hình.

### Dự đoán bệnh tim

- Nhập tay thông tin bệnh nhân.
- Upload file CSV/XLSX.
- Dự đoán hàng loạt nhiều bệnh nhân.
- Hiển thị kết quả dự đoán và xác suất nguy cơ.

### Giải thích kết quả

- Hiển thị các yếu tố làm tăng nguy cơ.
- Hiển thị các yếu tố làm giảm nguy cơ.
- Giải thích theo **rule-based logic** để người dùng dễ hiểu.

### Đánh giá mô hình

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Cài đặt môi trường

### Windows

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Train model

Chạy lệnh sau ở thư mục gốc của project:

```bash
python -m ml.train
```

---

## Fine-tune Random Forest

### Chạy nhanh để demo

```bash
python -m ml.grid_search_random_forest --fast --cv 3
```

### Chạy đầy đủ

```bash
python -m ml.grid_search_random_forest --cv 5
```

> Random Forest tự code sẽ chạy chậm hơn `sklearn`, nên khi demo có thể dùng chế độ `--fast`.

---

## Chạy Django

Di chuyển vào thư mục `web`:

```bash
cd web
python manage.py runserver
```

Mở trình duyệt và truy cập:

```text
http://127.0.0.1:8000/
```

---

## Dataset

Project sử dụng bộ dữ liệu:

**Heart Disease UCI Dataset**

Nguồn dữ liệu:

```text
https://archive.ics.uci.edu/ml/datasets/heart+Disease
```

Bộ dữ liệu gồm các đặc trưng y tế như tuổi, giới tính, loại đau ngực, huyết áp khi nghỉ, cholesterol, nhịp tim tối đa, oldpeak và một số chỉ số lâm sàng khác.  
Mục tiêu của mô hình là dự đoán bệnh nhân thuộc nhóm **không bệnh** hoặc **có nguy cơ mắc bệnh tim**.

---

## Ghi chú

- Project ưu tiên tính trực quan và khả năng demo.
- Phần model chính được tự code để hiểu rõ cách hoạt động của Random Forest.
- Phần giải thích kết quả sử dụng rule-based explanation để hỗ trợ người dùng hiểu kết quả dự đoán.
- Đây là project phục vụ học tập, không thay thế cho chẩn đoán y khoa thực tế.

---

## Tác giả

**Heart Disease Diagnosis - Manual Random Forest + Django**  
Project Machine Learning triển khai bằng Django.
