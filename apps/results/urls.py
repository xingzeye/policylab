from django.urls import path
from . import views

app_name = 'results'

urlpatterns = [
    path('', views.results_overview, name='results'),
    path('file/<str:algorithm>/<path:filename>/', views.result_file, name='result_file'),
]
