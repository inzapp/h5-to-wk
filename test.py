from glob import glob
from time import time

import caffe
import cv2
import numpy as np


input_shape=(416, 416)


model_path = './'

caffe_net = caffe.Net(model_path + '/model.prototxt', model_path + '/model.caffemodel', caffe.TEST)
#caffe_net.blobs['Placeholder'].reshape(1, 3, 16, 128)

cv_net = cv2.dnn.readNetFromCaffe(model_path + '/model.prototxt', model_path + '/model.caffemodel')

pb_net = cv2.dnn.readNet('frozen_graph.pb')


def main():
    print()
    print(caffe_net.blobs)
    print()

#    paths = glob('sample_images/*.jpg')
    paths = ['mapper/test_image.jpg']
    caffe_net_time_sum = 0
    cv_net_time_sum = 0
    for path in paths:
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (input_shape[1], input_shape[0]))
#        x = np.moveaxis(x, -1, 0)
#        x = np.asarray(x).astype('float32').reshape(1, 3, input_shape[0], input_shape[1]) / 255.
        x = np.asarray(x).astype('float32').reshape(1, 3, input_shape[0], input_shape[1])

        print(path)
        caffe_net_time_sum += inference_caffe_net(x)
        cv_net_time_sum += inference_cv_net(x)
        inference_pb_net(x)

    print(f'caffe net prediction time per frame : {(caffe_net_time_sum / len(paths)):.2f}')
    print(f'cv net prediction time per frame : {(cv_net_time_sum / len(paths)):.2f}')


def inference_caffe_net(x):
    caffe_net.blobs['Placeholder'].data[...] = x
    st = time()
    res = caffe_net.forward()
    et = time()
    res = res[list(res.keys())[0]]

    print('caffe net')
    print(f'shape : {res.shape}')
    channels = res.shape[1]
    rows = res.shape[2]
    cols = res.shape[3]
    for ch in range(channels):
        for i in range(rows):
            for j in range(cols):
               print('■' if res[0][ch][i][j] > 0.5 else '□', end='')
#               print(f'{(res[0][ch][i][j] * 127):.2f} ', end='')
            print()
        print()
    print()
    return et - st


def inference_cv_net(x):
    cv_net.setInput(x)
    st = time()
    res = cv_net.forward()
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
#                print(f'{(res[0][ch][i][j] * 127):.2f} ', end='')
            print()
        print()
    print()
    return et - st


def inference_pb_net(x):
    pb_net.setInput(x)
    st = time()
    res = pb_net.forward()
    et = time()

    print('pb dnn')
    print(f'shape : {res.shape}')
    channels = res.shape[1]
    rows = res.shape[2]
    cols = res.shape[3]
    for ch in range(channels):
        for i in range(rows):
            for j in range(cols):
                print('■' if res[0][ch][i][j] > 0.5 else '□', end='')
#                print(f'{(res[0][ch][i][j] * 127):.2f} ', end='')
            print()
        print()
    print()
    return et - st


def dnn_test():
    net = cv2.dnn.readNet('frozen_graph.pb')

    paths = glob('sample_images/*.jpg')
    for path in paths:
        print(path)
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (input_shape[1], input_shape[0]))
        x = np.moveaxis(x, -1, 0)
        x = np.asarray(x).astype('float32').reshape(1, 3, input_shape[0], input_shape[1]) / 255.
        net.setInput(x)
        res = net.forward()
        channels = res.shape[1]
        rows = res.shape[2]
        cols = res.shape[3]
        for ch in range(channels):
            for row in range(rows):
                for col in range(cols):
                    print('■' if res[0][ch][row][col] > 0.5 else '□', end='')
                print()
            print()
            print()
        pass


if __name__ == '__main__':
#    dnn_test()
    main()

