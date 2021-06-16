from django.urls import path
from . import views
app_name = 'process'
urlpatterns = [
    path('', views.objectCounting, name='objectCounting'),
    path('face-detection', views.faceDetection, name='faceDetection'),
    path('camera', views.camera, name='camera')
]