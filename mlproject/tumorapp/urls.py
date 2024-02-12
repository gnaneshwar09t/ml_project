from django.urls import path
from tumorapp import views

urlpatterns = [
path('', views.predict_tumor, name='predict_tumor'),
]