# -*- coding:utf-8 -*-

from django.conf.urls import url

from . import views


urlpatterns = [
    url(r"^recognition/(?P<image_id>[\w]+)$", views.ImagesRecognition.as_view()),
    url(r"^test$", views.Test.as_view())
]
