from django.urls import path
from . import views

urlpatterns = [
    path('segment/', views.segment_body, name='segment_body'),
]