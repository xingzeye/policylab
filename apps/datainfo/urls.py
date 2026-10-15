from django.urls import path
from . import views

app_name = 'datainfo'

urlpatterns = [
    path('', views.halfcheetah_overview, name='index'),
    path('halfcheetah-medium-v2/', views.halfcheetah_overview, name='halfcheetah_medium_v2'),
]
