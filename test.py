from glob import glob
from time import time

import caffe
import cv2
import numpy as np


input_shape=(368, 640)
color_mode = cv2.IMREAD_GRAYSCALE
input_channel = 3 if color_mode == cv2.IMREAD_COLOR else 1

model_path = './'

caffe_model = caffe.Net(model_path + '/model.prototxt', model_path + '/model.caffemodel', caffe.TEST)
caffe_dnn = cv2.dnn.readNetFromCaffe(model_path + '/model.prototxt', model_path + '/model.caffemodel')
tensorflow_dnn = cv2.dnn.readNet('model.pb')


def main():
    paths = glob('sample_images/*.jpg')
#    paths = ['mapper/test_image.jpg']
    caffe_model_time_sum = 0
    caffe_dnn_time_sum = 0
    tensorflow_dnn_time_sum = 0
    for path in paths:
        x = cv2.imread(path, color_mode)
        x = cv2.resize(x, (input_shape[1], input_shape[0]))
        x = np.moveaxis(x, -1, 0)
        x = np.asarray(x).astype('float32').reshape(1, input_channel, input_shape[0], input_shape[1]) / 255.

        print(path)
        caffe_model_time_sum += inference_caffe_model(x)
        caffe_dnn_time_sum += inference_caffe_dnn(x)
        tensorflow_dnn_time_sum += inference_tensorflow_dnn(x)

    print(f'caffe model forwarding time per frame : {(caffe_model_time_sum / len(paths)):.2f}')
    print(f'caffe dnn forwarding time per frame : {(caffe_dnn_time_sum / len(paths)):.2f}')
    print(f'tensorflow dnn forwarding time per frame : {(tensorflow_dnn_time_sum / len(paths)):.2f}')


def inference_caffe_model(x):
    caffe_model.blobs['Placeholder'].data[...] = x
    st = time()
    res = caffe_model.forward()
    et = time()
    res = res[list(res.keys())[0]]

    print('caffe model')
    print(f'shape : {res.shape}')
    channels = res.shape[1]
    rows = res.shape[2]
    cols = res.shape[3]
    for ch in range(channels):
        for i in range(rows):
            for j in range(cols):
               print('■' if res[0][ch][i][j] > 0.5 else '□', end='')
            print()
        print()
    print()
    return et - st


def inference_caffe_dnn(x):
    caffe_dnn.setInput(x)
    st = time()
    res = caffe_dnn.forward()
    et = time()

    print('caffe dnn')
    print(f'shape : {res.shape}')
    channels = res.shape[1]
    rows = res.shape[2]
    cols = res.shape[3]
    for ch in range(channels):
        for i in range(rows):
            for j in range(cols):
                print('■' if res[0][ch][i][j] > 0.5 else '□', end='')
            print()
        print()
    print()
    return et - st


def inference_tensorflow_dnn(x):
    tensorflow_dnn.setInput(x)
    st = time()
    res = tensorflow_dnn.forward()
    et = time()

    print('tensorflow dnn')
    print(f'shape : {res.shape}')
    channels = res.shape[1]
    rows = res.shape[2]
    cols = res.shape[3]
    for ch in range(channels):
        for i in range(rows):
            for j in range(cols):
                print('■' if res[0][ch][i][j] > 0.5 else '□', end='')
            print()
        print()
    print()
    return et - st


if __name__ == '__main__':
    main()

