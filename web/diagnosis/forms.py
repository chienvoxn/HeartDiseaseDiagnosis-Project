from django import forms


class HeartDiseaseForm(forms.Form):
    age = forms.IntegerField(label="Tuổi", min_value=1, max_value=120)

    sex = forms.ChoiceField(label="Giới tính", choices=[
        ("Male", "Nam"),
        ("Female", "Nữ"),
    ])

    cp = forms.ChoiceField(label="Loại đau ngực", choices=[
        ("typical angina", "Đau thắt ngực điển hình"),
        ("atypical angina", "Đau thắt ngực không điển hình"),
        ("non-anginal", "Đau ngực không do tim"),
        ("asymptomatic", "Không có triệu chứng"),
    ])

    trestbps = forms.FloatField(label="Huyết áp khi nghỉ", min_value=0)
    chol = forms.FloatField(label="Cholesterol trong máu", min_value=0)

    fbs = forms.ChoiceField(label="Đường huyết lúc đói > 120 mg/dl", choices=[
        ("False", "Không"),
        ("True", "Có"),
    ])

    restecg = forms.ChoiceField(label="Kết quả điện tâm đồ", choices=[
        ("normal", "Bình thường"),
        ("st-t abnormality", "Bất thường ST-T"),
        ("lv hypertrophy", "Phì đại thất trái"),
    ])

    thalch = forms.FloatField(label="Nhịp tim tối đa đạt được", min_value=0)

    exang = forms.ChoiceField(label="Đau thắt ngực khi vận động", choices=[
        ("False", "Không"),
        ("True", "Có"),
    ])

    oldpeak = forms.FloatField(label="Chỉ số ST chênh xuống khi vận động")

    slope = forms.ChoiceField(label="Độ dốc đoạn ST", choices=[
        ("upsloping", "Dốc lên"),
        ("flat", "Phẳng"),
        ("downsloping", "Dốc xuống"),
    ])

    ca = forms.ChoiceField(label="Số mạch máu chính được phát hiện", choices=[
        ("0", "0"),
        ("1", "1"),
        ("2", "2"),
        ("3", "3"),
        ("4", "4"),
    ])

    thal = forms.ChoiceField(label="Tình trạng Thalassemia", choices=[
        ("0", "Không rõ"),
        ("normal", "Bình thường"),
        ("fixed defect", "Khiếm khuyết cố định"),
        ("reversable defect", "Khiếm khuyết có thể đảo ngược"),
    ])

class CSVUploadForm(forms.Form):
    test_file = forms.FileField(
        label="File bệnh nhân cần dự đoán"
    )

    answer_file = forms.FileField(
        label="File đáp án để so sánh",
        required=False
    )

    def clean_test_file(self):
        file = self.cleaned_data["test_file"]

        if not (file.name.endswith(".csv") or file.name.endswith(".xlsx")):
            raise forms.ValidationError("Vui lòng tải lên file .csv hoặc .xlsx")

        return file

    def clean_answer_file(self):
        file = self.cleaned_data.get("answer_file")

        if file and not (file.name.endswith(".csv") or file.name.endswith(".xlsx")):
            raise forms.ValidationError("Vui lòng tải lên file .csv hoặc .xlsx")

        return file