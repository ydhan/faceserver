import os
import sys
import tensorflow as tf
from tensorflow.python.tools import saved_model_cli as cli
import argparse
import time
from tensorflow.python.saved_model import loader
from scipy import misc
import align.detect_face
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer

#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
#import visualization_utils as vis_util

class iface_api(object):
    """iface api.
    iface face detect, face predict api class

    """
    def __init__(self, model_name=None, saved_model_dir=None, tag_set='serve', signature_def_key='predict_images'):
        """class init function.
        """
        if model_name is not None or saved_model_dir is None:
            self.api_info = self.api_init_freeze_model(model_name)
        elif saved_model_dir != None:
            self.api_info = self.api_init_saved_model(saved_model_dir, tag_set, signature_def_key)
        self.mtcnn_info = self.mtcnn_init(self.api_info['sess'])

    #import saved model
    def api_init_saved_model(self, saved_model_dir, tag_set, signature_def_key):
        """init api use SavedModel.
        Args:
            saved_model_dir: Directory containing the SavedModel to load.
            tag_set: Group of tag(s) of the MetaGraphDef with the SignatureDef map, in
                string format, separated by ','. For tag-set contains multiple tags, all
                tags must be passed in.
            signature_def_key: A SignatureDef key string.
        Returns:
            api_info: a dict store session and model input output info
            api_info = {'sess': sess,
                        'inputs_dict': inputs_feed_dict,
                        'outputs_dict': outputs_return_dict}
        Raises:
            RuntimeError: An error when model file not exists
        """
        # Get a list of output tensor names.
        meta_graph_def = cli.get_meta_graph_def(saved_model_dir, tag_set)

        # Re-create feed_dict based on input tensor name instead of key as session.run
        # uses tensor name.
        inputs_tensor_info = cli._get_inputs_tensor_info_from_meta_graph_def(
            meta_graph_def, signature_def_key)
        inputs_feed_dict = {
            #key:outputs_tensor_info[key].name: tensor
            key: tensor.name
            for key, tensor in inputs_tensor_info.items()
        }
        # Get outputs
        outputs_tensor_info = cli._get_outputs_tensor_info_from_meta_graph_def(
            meta_graph_def, signature_def_key)
        outputs_return_dict = {
            #key:outputs_tensor_info[key].name: tensor
            key: tensor.name
            for key, tensor in outputs_tensor_info.items()
        }

        outputs_return_dict['embeddings'] = 'embeddings:0'
        #print('output return dict:', outputs_return_dict)

        #graph = tf.Graph()
        #graph.as_default()
        #sess = tf.Session(graph=graph)
        sess = tf.Session()
        #sess = tf.Session(graph=ops_lib.Graph())
        loader.load(sess, tag_set.split(','), saved_model_dir)
        #print_info()
        api_info = {'sess': sess,
                    'inputs_dict': inputs_feed_dict,
                    'outputs_dict': outputs_return_dict}

        #print('api_info:', api_info)
        return api_info

    # import graph
    def api_init_freeze_model(self, model_file):
        """init api use freeze model.
        Args:
            model_file: freeze model file name(.pb) to load.
        Returns:
            api_info: a dict store session and model input output info
            api_info = {'sess': sess,
                        'inputs_dict': inputs_feed_dict,
                        'outputs_dict': outputs_return_dict}
        Raises:
            RuntimeError: An error when modle file not exists.
        """
        if model_file is None:
            model_dir,_ = os.path.split(os.path.realpath(__file__))
            model_file = os.path.join(model_dir, 'model', 'iface_freeze_model.pb')

        # Get a list of output tensor names.
        output_graph_def = graph_pb2.GraphDef()
        with open(model_file, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            #_ = importer.import_graph_def(output_graph_def, name="")
            tf.import_graph_def(output_graph_def, name="")

        outputs_return_dict = {'class': 'dnn/predictions/class_string_lookup_Lookup:0',
                            'scores': 'dnn/predictions/predictions_reduce_max:0',
                               'embeddings': 'embeddings:0'}
        inputs_feed_dict = {'images': 'images_input:0'}

        sess = tf.Session()
        sess.run("init_all_tables")
        api_info = {'sess': sess,
                    'inputs_dict': inputs_feed_dict,
                    'outputs_dict': outputs_return_dict}

        return api_info

    def predi_with_feed(self, input_tensor_key_feed_dict):
        """run face prediction graph, and get classname and scores.

        Args:
            input_tensor_key_feed_dict: A dictionary maps input keys to numpy ndarrays.
        Returns:
            a dict hold class and scores returned by predict model
            outputs = {
                        'class': classname string array,
                        'scores': scores float array}
        Raises:
            RuntimeError: An error when output file already exists and overwrite is not
            enabled.
        """
        # Re-create feed_dict based on input tensor name instead of key as session.run
        # uses tensor name.
        inputs_feed_dict = self.api_info['inputs_dict']

        inputs_feed_dict = {
            tensor: input_tensor_key_feed_dict[key]
            for key, tensor in inputs_feed_dict.items()
        }
        #print('input dict:', inputs_feed_dict)
        # Get outputs
        sess = self.api_info['sess']
        t = time.time()
        outputs = sess.run(self.api_info['outputs_dict'], feed_dict=inputs_feed_dict)
        print('run predi with dict time:', time.time()-t)
        return outputs

    def predi_face_image(self, images):
        """predict face use numpy uint8 image."""

        outputs = self.predi_with_feed({'images':images})
        return outputs

    def predi_face(self, image_path):
        """predict face use image file name."""

        if os.path.exists(image_path):
            try:
                img = misc.imread(image_path)
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(image_path, e)
                print(errorMessage)
            else:
                return self.predi_face_image([img])

    def mtcnn_init(self, sess):
        """mtcnn model init.
        """
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
        mtcnn_info = {'pnet': pnet, 'rnet': rnet, 'onet': onet}
        return mtcnn_info

    def detect_face_image0(self, img, image_size=160, margin=40):
        """run  mtcnn, and get one face bounding boxes.

        Args:
            img: image of uint8 h*w*3 numpy ndarrays.
            image_size: size of face to resize to
            margin: margin width to return face
        Returns:
            a dict hold detected face boxes, resized face image array
            outputs = {
                        'box': face bounding box array, sharp is 4, each element is ymin, xmin, ymax, xmax,
                        'cropped': cropped face
                        'scaled': scaled face image}
        Raises:
            RuntimeError: An error when output file already exists and overwrite is not
            enabled.
        """
        minsize = 20  # minimum size of face
        threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        factor = 0.709  # scale factor
        if img.ndim<3:
            return
        pnet = self.mtcnn_info['pnet']
        rnet = self.mtcnn_info['rnet']
        onet = self.mtcnn_info['onet']
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        #print('found %s face' % nrof_faces)
        #print('boxs:', bounding_boxes)
        if nrof_faces>0:
            det = bounding_boxes[:,0:4]
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces>1:
                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                det = det[index,:]
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            #scaled = facenet.prewhiten(scaled)
            bb2 = np.zeros(4, dtype=np.int32)
            bb2[0] = bb[1]
            bb2[1] = bb[0]
            bb2[2] = bb[3]
            bb2[3] = bb[2]
            return {'box':bb2, 'cropped':cropped, 'scaled':scaled}
        else:
            print('Unable detect face')

    def detect_face_image(self, img, image_size=160, margin=40):
        """run  mtcnn, and get all face bounding boxes.

        Args:
            img: image of uint8 h*w*3 numpy ndarrays.
            image_size: size of face to resize to
            margin: margin width to return face
        Returns:
            a dict hold detected face boxes, resized face image array
            outputs = {
                        'boxes': face bounding box array, sharp is N*4, each element is ymin, xmin, ymax, xmax,
                        'scaled': scaled face image array}
        Raises:
            RuntimeError: An error when output file already exists and overwrite is not
            enabled.
        """
        minsize = 20  # minimum size of face
        threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        factor = 0.709  # scale factor
        if img.ndim<3:
            return
        pnet = self.mtcnn_info['pnet']
        rnet = self.mtcnn_info['rnet']
        onet = self.mtcnn_info['onet']
        t = time.time()
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        print('mtcnn detect face time:', time.time()-t)
        nrof_faces = bounding_boxes.shape[0]
        #print('found %s face' % nrof_faces)
        #print('boxs:', bounding_boxes)
        if nrof_faces>0:
            det = bounding_boxes[:,0:4]
            img_size = np.asarray(img.shape)[0:2]
            bb = np.zeros([nrof_faces,4], dtype=np.int32)
            scaled = np.zeros([nrof_faces,image_size,image_size,3], dtype=np.uint8)
            margin_rate = float(margin)/image_size
            for i in range(nrof_faces):
                bb[i, 1] = np.maximum(det[i,0]-(det[i,2]-det[i,0])*margin_rate/2, 0)
                bb[i, 0] = np.maximum(det[i,1]-(det[i,3]-det[i,1])*margin_rate/2, 0)
                bb[i, 3] = np.minimum(det[i,2]+(det[i,2]-det[i,0])*margin_rate/2, img_size[1])
                bb[i, 2] = np.minimum(det[i,3]+(det[i,3]-det[i,1])*margin_rate/2, img_size[0])
                cropped = img[bb[i,0]:bb[i,2],bb[i,1]:bb[i,3],:]
                scaled[i] = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                #scaled = facenet.prewhiten(scaled)
            #bb[0:4] ymin,xmin,ymax,xmax
            return {'boxes':bb, 'scaled':scaled}
        else:
            print('Unable detect face')

    def detect_face(self, image_path, image_size=160, margin=40):
        """detect face by image file name."""

        if  os.path.exists(image_path):
            try:
                img = misc.imread(image_path)
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(image_path, e)
                print(errorMessage)
            else:
                return self.detect_face_image(img, image_size, margin)

    def detect_predi_face(self, image_path, image_size=160, margin=40):
        """detect face and predict it using image file name."""

        detect_res = self.detect_face(image_path, image_size, margin)
        if detect_res is not None:
            predi_res = self.predi_face_image(detect_res['scaled'])
            return {'boxes':detect_res['boxes'], 'class':predi_res['class'],
                    'scores':predi_res['scores'], 'scaled_face':detect_res['scaled']}

    def detect_predi_face_image(self, image, image_size=160, margin=40):
        """detect face and predict it using numpy uint8 image."""

        detect_res = self.detect_face_image(image, image_size, margin)
        if detect_res is not None:
            predi_res = self.predi_face_image(detect_res['scaled'])
            return {'boxes':detect_res['boxes'], 'class':predi_res['class'],
                    'scores':predi_res['scores'], 'scaled_face':detect_res['scaled']}

    def identity_face_image(self, image1, image2, threshold=0.8):
        """give two image, identity if they are the same person.

        Args:
            image1: first image of person
            image2: second image of person
            threshold: threshold to disinguish from two persion

        Returns:
            Ture if the twom image is from one persion, otherwise reutn False
        """
        image_size = 160
        margin = 40
        detect_res1 = self.detect_face_image(image1, image_size, margin)
        if detect_res1 is not None:
            detect_res2 = self.detect_face_image(image2, image_size, margin)
            if detect_res2 is not None:
                print('image1:', detect_res1['scaled'][0].shape)
                print('image2:', detect_res2['scaled'][0].shape)
                predi_res = self.predi_face_image(np.stack((detect_res1['scaled'][0], detect_res2['scaled'][0])))
                diff = np.subtract(predi_res['embeddings'][0], predi_res['embeddings'][1])
                dist = np.sum(np.square(diff))
                print('dist:', dist, 'threshold:', threshold)
                return dist<threshold, dist

    def identity_face(self, image_file1, image_file2, threshold=0.8):
        """give two image file, identity if they are the same person.

        Args:
            image_file1: first image file name of person
            image_file22: second image file name of person
            threshold: threshold to disinguish from two persion

        Returns:
            Ture if the twom image is from one persion, otherwise reutn False
        """
        image_size = 160
        margin = 40
        detect_res1 = self.detect_face(image_file1, image_size, margin)
        if detect_res1 is not None:
            detect_res2 = self.detect_face(image_file2, image_size, margin)
            if detect_res2 is not None:
                predi_res = self.predi_face_image(np.stack((detect_res1['scaled'][0], detect_res2['scaled'][0])))
                diff = np.subtract(predi_res['embeddings'][0], predi_res['embeddings'][1])
                dist = np.sum(np.square(diff))
                print('dist:', dist, 'threshold:', threshold)
                return dist<threshold, dist

    def face_feature_vector(self, image_file):
        """image file, extract feature vector.

        Args:
            image_file: image file name of face

        Returns:
            image feature vector, length is 128
        """
        if  os.path.exists(image_file):
            try:
                img = misc.imread(image_file)
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(image_path, e)
                print(errorMessage)
            else:
                image_size = 160
                if img.shape[0] != image_size or img.shape[1] != image_size:
                    img = misc.imresize(img, (image_size, image_size), interp='bilinear')
                predi_res = self.predi_face_image([img])
                return predi_res['embeddings'][0]

    def face_image_feature_vector(self, images):
        """images, extract feature vector.

        Args:
            image: images of face, shape must be n*160*160*3

        Returns:
            image feature vector, length is 128
        """
        image_size = 160
        if images[0].shape[0] != image_size or images[0].shape[1] != image_size:
            return
        predi_res = self.predi_face_image(images)
        return predi_res['embeddings']

