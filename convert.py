#!/usr/bin/env python2.7
# coding = utf-8
# pylint: disable=invalid-name, no-member
'''This script convert the G model for Mini-Caffe
'''
import cv2
import caffe
import mxnet as mx
import numpy as np


def convert():
    '''convert mxnet parameters to caffe parameters
    '''
    net = caffe.Net('g.prototxt', caffe.TRAIN)
    caffe_params = net.params
    mx_params = mx.nd.load('model/face-G-0000.params')
    # 1x1
    caffe_params['gconv1'][0].data[...] = mx_params['arg:gconv1_weight'].asnumpy()
    caffe_params['gbn1'][0].data[...] = mx_params['aux:gbn1_moving_mean'].asnumpy()
    caffe_params['gbn1'][1].data[...] = mx_params['aux:gbn1_moving_var'].asnumpy()
    caffe_params['gbn1_scale'][0].data[...] = mx_params['arg:gbn1_gamma'].asnumpy()
    caffe_params['gbn1_scale'][1].data[...] = mx_params['arg:gbn1_beta'].asnumpy()
    # 4x4
    caffe_params['gconv2'][0].data[...] = mx_params['arg:gconv2_weight'].asnumpy()
    caffe_params['gbn2'][0].data[...] = mx_params['aux:gbn2_moving_mean'].asnumpy()
    caffe_params['gbn2'][1].data[...] = mx_params['aux:gbn2_moving_var'].asnumpy()
    caffe_params['gbn2_scale'][0].data[...] = mx_params['arg:gbn2_gamma'].asnumpy()
    caffe_params['gbn2_scale'][1].data[...] = mx_params['arg:gbn2_beta'].asnumpy()
    # 8x8
    caffe_params['gconv3'][0].data[...] = mx_params['arg:gconv3_weight'].asnumpy()
    caffe_params['gbn3'][0].data[...] = mx_params['aux:gbn3_moving_mean'].asnumpy()
    caffe_params['gbn3'][1].data[...] = mx_params['aux:gbn3_moving_var'].asnumpy()
    caffe_params['gbn3_scale'][0].data[...] = mx_params['arg:gbn3_gamma'].asnumpy()
    caffe_params['gbn3_scale'][1].data[...] = mx_params['arg:gbn3_beta'].asnumpy()
    # 16x16
    caffe_params['gconv4'][0].data[...] = mx_params['arg:gconv4_weight'].asnumpy()
    caffe_params['gbn4'][0].data[...] = mx_params['aux:gbn4_moving_mean'].asnumpy()
    caffe_params['gbn4'][1].data[...] = mx_params['aux:gbn4_moving_var'].asnumpy()
    caffe_params['gbn4_scale'][0].data[...] = mx_params['arg:gbn4_gamma'].asnumpy()
    caffe_params['gbn4_scale'][1].data[...] = mx_params['arg:gbn4_beta'].asnumpy()
    # 32x32
    caffe_params['gconv5'][0].data[...] = mx_params['arg:gconv5_weight'].asnumpy()
    return net


def visual(fn, data):
    '''visualize data to fn
    '''
    assert len(data.shape) == 4
    data = data.transpose((0, 2, 3, 1))
    data = data * 127.5 + 127.5
    data = np.clip(data, 0, 255).astype(np.uint8)
    n = int(np.ceil(np.sqrt(data.shape[0])))
    h, w, c = data.shape[1:]
    canvas = np.zeros((n*h, n*w, c), dtype=np.uint8)
    for i in range(data.shape[0]):
        y, x = i / n, i % n
        canvas[y*h:y*h+h, x*w:x*w+w, :] = data[i]
    # save canvas
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imwrite(fn, canvas)


def test():
    '''test caffe_net and mx_net
    '''
    noise = np.random.normal(0, 1, (64, 100, 1, 1))
    caffe_net.blobs['data'].data[...] = noise
    caffe_net.forward()
    caffe_image = caffe_net.blobs['gconv5'].data
    mx_net.forward(mx.io.DataBatch([mx.nd.array(noise)], []), is_train=False)
    mx_image = mx_net.get_outputs()[0].asnumpy()
    visual('tmp/caffe-g.png', caffe_image)
    visual('tmp/mxnet-g.png', mx_image)
    error = caffe_image - mx_image
    error = np.square(error).mean()
    print 'error:', error


if __name__ == '__main__':
    caffe_net = convert()
    mx_net = mx.mod.Module.load('model/face-G', 0)
    mx_net.bind(data_shapes=[('rand', (64, 100, 1, 1))])
    test()
    caffe_net.save('model/g.caffemodel')
