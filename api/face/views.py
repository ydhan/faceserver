# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
from django.shortcuts import render
from django.http import HttpResponse
from rest_framework import viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
import ifaceapi.iface_api as ia
import Image
from scipy import misc
import numpy as np
import ifaceapi.visualization_utils as vis_util


# Create your views here.

gflag = 1
gface1name = "f-1.jpg"
gface2name = "f-2.jpg"

iface = ia.iface_api()


def downloadimg(img):
    #name = "./static/images/img.jpg"
    name ="./static/images/"+img._get_name()
    
    f = open(name,'wb')
    for chunk in img.chunks():
        f.write(chunk)
    f.close()

def detect_image(name):
    image_path=name
    for i in range(1):
        print('start to detect')
        #res = iface.detect_face(image_path)
        #outputs = iface.predi_face_image(res['scaled'])
        #outputs=iface.detect_predi_face(image_path)
        img=misc.imread(image_path)
        outputs=iface.detect_predi_face_image(img)
        if outputs is None:
            print('failed to detect face')
            return
        #print('detect and predi result:', outputs)
        predi_class = outputs['class']
        predi_scores = outputs['scores']
        bb=outputs['boxes']
        print('detect and predi result:', predi_class, predi_scores, bb)
        label=[]
        for i in range(len(predi_class)):
            label.append(['%s,%f'%(predi_class[i], predi_scores[i])])
        bbs=bb.astype(np.float)
        bbs[:,0::2] /= img.shape[0]
        bbs[:,1::2] /= img.shape[1]
        vis_util.draw_bounding_boxes_on_image_array(img, bbs, color='red',
            thickness=4, display_str_list_list=label)

        _, name = os.path.split(image_path)
        base, ext = os.path.splitext(name)
        vis_util.save_image_array_as_png(img, os.path.join('./static/images', base+'_detected'+ext))
        print('saved to', os.path.join('./static/images', base+'_detected'+ext))

class FaceViewSet(APIView):
    def get(self, request, *args, **kwargs):
        return Response('GET')
 
    def post(self, request, *args, **kwargs):
        global gflag
        global gface1name
        global gface2name
        print(request)
        print(request.get_full_path()) 
        #print(request.body) 
        #print(request.path) 
        print(request.FILES) 
        #print(request.data)
        img=request.FILES['upload']
        print(img._get_name())
        if gflag == 1:
            gface1name = img._get_name()
            gflag=0
        else:
            gface2name = img._get_name()
            gflag=1
        #print(img[TemporaryUploadedFile])
        downloadimg(img) 
        #detect_image("./static/images/img.jpg")
        #print(request.META) 
        #print(request.session) 
        #print(request.resolver_match) 
        #return Response('POST')
        return Response('http://192.168.0.156:8000/static/images/'+img._get_name())
        #return Response('http://192.168.0.156:8000/static/images/img.jpg')
 
    def put(self, request, *args, **kwargs):
        return Response('PUT')
    
def detect(request):
    print(request)
    detect_image("./static/images/"+gface1name)
    base, ext = os.path.splitext(gface1name)
    return HttpResponse('http://192.168.0.156:8000/static/images/'+base+'_detected'+ext)

def predict(request):
    return HttpResponse("predict image")

def compare(request):
    print('./static/images/'+gface1name)
    print('./static/images/'+gface2name)
    same, val=iface.identity_face('./static/images/'+gface1name, './static/images/'+gface2name, 0.8)
    str = ("距离小于0.8判断为同一个人，其距离为:%f,是否同一个人:%d"%(val,same))    
    return HttpResponse(str)

def find(request):
    return HttpResponse("find image")
 
