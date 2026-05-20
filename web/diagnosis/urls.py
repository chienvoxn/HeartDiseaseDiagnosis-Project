from django.urls import path
from . import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Trang chính
    path("", views.diagnosis_view, name="diagnosis"),
    path("dashboard/", views.dashboard_view, name="dashboard"),
    path("predict/", views.predict_view, name="predict"),
    path("evaluation/", views.evaluation_view, name="evaluation"),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)