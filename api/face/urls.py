# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.conf.urls import include, url
from . import views
from views import FaceViewSet

urlpatterns = [ 
    url(r'^$', FaceViewSet.as_view()),
    url(r'^detect/',views.detect,name = 'detect'),
    url(r'^compare/',views.compare,name = 'compare'),
]
