Heart Disease Diagnosis - Manual Random Forest + Django
Project dự đoán nguy cơ bệnh tim dựa trên bộ dữ liệu Heart Disease UCI.
Mục tiêu
Xây dựng pipeline Machine Learning rõ ràng.
Không dùng `scikit-learn` cho phần train model chính.
Model chính: Random Forest tự code.
Fine-tuning bằng Manual Grid Search.
Deploy local bằng Django.
Giao diện dashboard trực quan cho dự đoán và đánh giá kết quả.
Kết hợp giải thích dự đoán theo rule-based logic.
---
Công nghệ sử dụng
Machine Learning
Python
NumPy
Pandas
Manual Decision Tree
Manual Random Forest
Manual Grid Search
Pickle
Web Deployment
Django
HTML/CSS
Django Template Engine
---
Lưu ý về thư viện
Các thành phần sau được tự code, không dùng `scikit-learn`:
preprocessing pipeline
train/test split
metrics
Decision Tree
Random Forest
Grid Search
prediction pipeline
---
Cấu trúc project
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
│   │
│   ├── reports/
│   │
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
│   │
│   ├── heart_project/
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   │
│   ├── media/
│   │
│   └── diagnosis/
│       ├── static/
│       │   └── diagnosis/
│       │       └── style.css
│       │
│       ├── templates/
│       │   └── diagnosis/
│       │       ├── base_dashboard.html
│       │       ├── dashboard.html
│       │       ├── evaluation.html
│       │       ├── index.html
│       │       ├── login.html
│       │       └── predict.html
│       │
│       ├── templatetags/
│       │   ├── __init__.py
│       │   └── custom_filters.py
│       │
│       ├── forms.py
│       ├── views.py
│       ├── urls.py
│       └── apps.py
│
├── requirements.txt
└── README.md
```
---
Pipeline Machine Learning
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
Chức năng hệ thống
Trang chủ
Giới thiệu hệ thống dự đoán bệnh tim.
Hiển thị thông tin mô hình và dữ liệu.
Dashboard
Tổng quan dataset.
Số lượng bệnh nhân.
Thống kê nguy cơ bệnh tim.
Dự đoán bệnh tim
Nhập tay thông tin bệnh nhân.
Upload file CSV/XLSX.
Dự đoán hàng loạt nhiều bệnh nhân.
Giải thích kết quả
Hiển thị các yếu tố làm tăng nguy cơ.
Hiển thị các yếu tố làm giảm nguy cơ.
Giải thích theo rule-based logic.
Đánh giá mô hình
Accuracy
Precision
Recall
F1-score
Confusion Matrix
---
Cài đặt môi trường
Windows
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
---
Train model
```bash
python -m ml.train
```
---
Fine-tune Random Forest
Chạy nhanh:
```bash
python -m ml.grid_search_random_forest --fast --cv 3
```
Chạy đầy đủ:
```bash
python -m ml.grid_search_random_forest --cv 5
```
Lưu ý:
Random Forest tự code sẽ chậm hơn `sklearn`.
Khi demo nên dùng `--fast`.
---
Chạy Django
```bash
cd web
python manage.py runserver
```
Mở trình duyệt:
```text
http://127.0.0.1:8000/
```
---
Dataset
Sử dụng:
Heart Disease UCI Dataset
Nguồn:
https://archive.ics.uci.edu/ml/datasets/heart+Disease
---
Ghi chú
Project tập trung vào việc hiểu rõ cách hoạt động của Random Forest.
Ưu tiên tính trực quan và khả năng demo.
Phần giải thích kết quả sử dụng rule-based explanation để hỗ trợ người dùng hiểu kết quả dự đoán.