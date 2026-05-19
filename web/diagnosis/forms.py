from django import forms


class HeartDiseaseForm(forms.Form):
    age = forms.IntegerField(label="Age", min_value=1, max_value=120)
    sex = forms.ChoiceField(label="Sex", choices=[("Male", "Male"), ("Female", "Female")])
    cp = forms.ChoiceField(label="Chest Pain Type", choices=[
        ("typical angina", "Typical angina"),
        ("atypical angina", "Atypical angina"),
        ("non-anginal", "Non-anginal"),
        ("asymptomatic", "Asymptomatic"),
    ])
    trestbps = forms.FloatField(label="Resting Blood Pressure", min_value=0)
    chol = forms.FloatField(label="Cholesterol", min_value=0)
    fbs = forms.ChoiceField(label="Fasting Blood Sugar > 120", choices=[("False", "False"), ("True", "True")])
    restecg = forms.ChoiceField(label="Resting ECG", choices=[
        ("normal", "Normal"),
        ("stt abnormality", "ST-T abnormality"),
        ("lv hypertrophy", "LV hypertrophy"),
    ])
    thalch = forms.FloatField(label="Maximum Heart Rate", min_value=0)
    exang = forms.ChoiceField(label="Exercise Induced Angina", choices=[("False", "False"), ("True", "True")])
    oldpeak = forms.FloatField(label="Oldpeak")
    slope = forms.ChoiceField(label="Slope", choices=[
        ("upsloping", "Upsloping"),
        ("flat", "Flat"),
        ("downsloping", "Downsloping"),
    ])
    ca = forms.ChoiceField(label="Number of Major Vessels", choices=[("0", "0"), ("1", "1"), ("2", "2"), ("3", "3")])
    thal = forms.ChoiceField(label="Thal", choices=[
        ("normal", "Normal"),
        ("fixed defect", "Fixed defect"),
        ("reversable defect", "Reversable defect"),
    ])
