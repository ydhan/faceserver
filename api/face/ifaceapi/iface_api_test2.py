import os
import sys
import tensorflow as tf
from tensorflow.python.tools import saved_model_cli as cli
import argparse
import time
from tensorflow.python.saved_model import loader
from scipy import misc
#import ifaceapi.align.detect_face
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
#from ifaceapi import iface_api as ia
import iface_api as ia

#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
import visualization_utils as vis_util


def detect_test(args):
    
    #image_path='test_image/AaronEckhart_11_11.jpg'
    image_path=args.filename
    #image_path='/raid/download/Fscrub-align/AdamBrody'
    iface = ia.iface_api()

    for i in range(1):
        #res = iface.detect_face(image_path)
        #outputs = iface.predi_face_image(res['scaled'])
        #outputs=iface.detect_predi_face(image_path)
        img=misc.imread(image_path)
        outputs=iface.detect_predi_face_image(img)
        if outputs is None:
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
        vis_util.save_image_array_as_png(img, os.path.join('./out_img', base+'_detected'+ext))
        print('saved to', os.path.join('./out_img', base+'_detected'+ext))

def api_test():
    model_dir='test_model/dnn_model/export/1'
    model_name='test_model/dnn_model/iface_freeze_model.pb'
    #image_path='test_image2/AaronEckhart_10_10.png'
    image_path='/raid/download/Fscrub-align/AdamBrody'
    print('before init api', time.time())
    iface = iface_api(model_name)
    print('after init api', time.time())

    #print('before read img', time.time())
    #img = misc.imread(image_path)
    #print('after read img', time.time())
    #face_predi([img])

    #for f in os.listdir(image_path):
    #    print('before predi img', time.time())
    #    img = misc.imread(os.path.join(image_path, f))
    #    outputs = iface.predi([img])
    #    print('after predi img', time.time())
    #    print('class:%s, probi:%f' % (outputs['class'][0], outputs['scores'][0]))
    images=[]
    for f in os.listdir(image_path):
        img = misc.imread(os.path.join(image_path, f))
        images.append(img)

    print('image len:', len(images))
    print('before predi img', time.time())
    outputs = iface.predi_face_image(images)
    print('after predi img', time.time())

    print('before predi img', time.time())
    outputs = iface.predi_face_image(images)
    print('after predi img', time.time())
    #print('outputs:', outputs)

    for i in range(len(images)):
        print('class:%s, probi:%f' % (outputs['class'][i], outputs['scores'][i]))

#api_test()
#detect_test()

def print_info():
    #tvar = tf.trainable_variables()
    tvar = tf.global_variables()

    print('global variable num:', len(tvar))
    #tvar_name = [x.name for x in tvar]
    for x in tvar:
        print('name:', x.name)
        print('var:', x)

    #return
    g = tf.get_default_graph()
    ops  = g.get_operations()
    print('opt variable num:', len(ops))
    for (i, op) in enumerate(ops):
        if not (op.name.startswith('InceptionResnetV1')
            or  op.name.startswith('save')
            or  op.name.startswith('restore')):
            print(i, op.name, op)

    return
    nodes = g.as_graph_def().node
    print('node len', len(nodes))
    for node in nodes:
        if not node.name.startswith('InceptionResnetV1'):
            print(node.name, op.op)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', type=str,
        help='file name')

    #parser.add_argument('--model_name', type=str,
    #    help='file name')
    return parser.parse_args(argv)

if __name__ == '__main__':
    detect_test(parse_arguments(sys.argv[1:]))
