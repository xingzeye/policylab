from django.urls import path
from . import views

app_name = 'modelsinfo'

urlpatterns = [
    path('', views.modelsinfo, name='modelsinfo'),
]
