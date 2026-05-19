# Heart Disease Diagnosis - Manual Random Forest + Django + SHAP

Project dự đoán nguy cơ bệnh tim dựa trên bộ dữ liệu Heart Disease UCI.

## Mục tiêu

- Xây dựng pipeline Machine Learning rõ ràng.
- Không dùng `scikit-learn` cho phần train model chính.
- Model chính: Random Forest tự code.
- Fine-tuning: Manual Grid Search giống notebook.
- Deploy local bằng Django.
- Có SHAP để giải thích tầm quan trọng của feature.

## Lưu ý về thư viện

Phần sau **không dùng scikit-learn**:

- preprocessing pipeline
- train/test split
- metrics
- Decision Tree
- Random Forest
- Grid Search
- prediction trong Django

Phần SHAP được phép dùng thư viện:

- `shap`
- các dependency hỗ trợ của SHAP như `scikit-learn` nếu thư viện cần

SHAP chỉ dùng cho bước Explainable AI sau khi model đã train, không tham gia vào quá trình huấn luyện model.

## Cấu trúc project

```text
HeartDisease_RF_Django_NoSklearn_v4_SHAP_Library/
│
├── data/
│   └── heart_disease_uci.csv
│
├── notebooks/
│   └── HeartDiseaseDiagnosis.ipynb
│
├── ml/
│   ├── preprocessing.py
│   ├── tree.py
│   ├── random_forest.py
│   ├── pipeline.py
│   ├── metrics.py
│   ├── data_utils.py
│   ├── train.py
│   ├── grid_search_random_forest.py
│   ├── shap_explain.py
│   ├── predict.py
│   ├── models/
│   │   └── manual_random_forest.pkl
│   └── reports/
│       ├── metrics.json
│       ├── tuning_report.json
│       ├── shap_summary_bar.png
│       ├── shap_summary_dot.png
│       ├── shap_feature_importance.csv
│       └── shap_report.json
│
├── web/
│   ├── manage.py
│   ├── heart_project/
│   └── diagnosis/
│       ├── forms.py
│       ├── views.py
│       ├── urls.py
│       └── templates/
│           └── diagnosis/
│               ├── index.html
│               └── explanation.html
│
├── requirements.txt
└── README.md
```

## Pipeline ML

```text
Raw data
→ tạo target binary từ num
→ bỏ id, dataset, num, target
→ chuyển trestbps = 0 và chol = 0 thành NaN
→ stratified train/test split
→ preprocessing tự code:
   - numeric: median imputation + standard scaling
   - categorical: mode imputation + one-hot encoding
→ Manual Random Forest
→ evaluate bằng Accuracy, Precision, Recall, F1-score
→ Manual Grid Search để fine-tune Random Forest
→ lưu model bằng pickle
→ Django load model để predict
→ SHAP giải thích feature importance
```

## Cài đặt

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Trên macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train nhanh model

```bash
python -m ml.train
```

## Fine-tune Random Forest bằng Grid Search

Chạy nhanh để test project:

```bash
python -m ml.grid_search_random_forest --fast --cv 3
```

Chạy giống hướng notebook hơn:

```bash
python -m ml.grid_search_random_forest --cv 5
```

Lưu ý: bản Random Forest tự code sẽ chậm hơn `sklearn`, nên khi demo nên dùng `--fast` trước.

## Chạy SHAP

Sau khi train hoặc grid search xong, chạy:

```bash
python -m ml.shap_explain --background-size 30 --sample-size 40 --nsamples 80
```

Output:

```text
ml/reports/shap_summary_bar.png
ml/reports/shap_summary_dot.png
ml/reports/shap_feature_importance.csv
ml/reports/shap_report.json
```

Script cũng copy ảnh SHAP sang:

```text
web/diagnosis/static/diagnosis/images/
```

để Django hiển thị ở trang:

```text
http://127.0.0.1:8000/explanation/
```

## Chạy Django

```bash
cd web
python manage.py runserver
```

Mở:

```text
http://127.0.0.1:8000/
```

Trang giải thích SHAP:

```text
http://127.0.0.1:8000/explanation/
```

## SHAP có cần không?

Không bắt buộc để model dự đoán. Nhưng SHAP giúp project tốt hơn vì:

- giải thích feature nào quan trọng nhất
- cho thấy feature làm tăng hay giảm nguy cơ bệnh tim
- phù hợp với bài toán y tế vì cần tính giải thích
- làm báo cáo chuyên nghiệp hơn

Trong project này, SHAP được dùng cho phân tích sau huấn luyện, không dùng trong quá trình train.
