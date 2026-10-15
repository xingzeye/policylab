
from django.urls import path
from . import views

app_name = 'compare'

urlpatterns = [
    path('', views.compare_view, name='compare'),
    path('asset/<path:relative_path>/', views.compare_asset, name='asset'),
]
