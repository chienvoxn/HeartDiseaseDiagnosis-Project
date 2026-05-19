from django.urls import path
from . import views

urlpatterns = [
    path("", views.diagnosis_view, name="diagnosis"),
    path("explanation/", views.explanation_view, name="explanation"),
]
