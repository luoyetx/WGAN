#!/usr/bin/env python2.7
# coding = utf-8
# pylint: disable=invalid-name, no-member, too-many-arguments, line-too-long
from __future__ import print_function
import logging
import argparse
import cv2
import mxnet as mx
import numpy as np


logging.basicConfig(format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


########## model part ##########

def dcgan64x64(ngf, ndf, nc, eps=2e-5):
    '''dcgan with output size 64x64

    Parameters
    ----------
    ngf: base filter number of generator
    ndf: base filter number of discriminator
    nc: number of channels for generator output
    eps: eps for BatchNorm

    Return
    ------
    G: generator
    D: discriminator
    '''
    # generator
    rand = mx.sym.Variable('rand')
    # 1x1
    gconv1 = mx.sym.Deconvolution(rand, name='gconv1', kernel=(4, 4), num_filter=ngf*8, no_bias=True)
    gbn1 = mx.sym.BatchNorm(gconv1, name='gbn1', fix_gamma=True, eps=eps)
    gact1 = mx.sym.Activation(gbn1, name='gact1', act_type='relu')
    # 4x4
    gconv2 = mx.sym.Deconvolution(gact1, name='gconv2', kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                                  num_filter=ngf*4, no_bias=True)
    gbn2 = mx.sym.BatchNorm(gconv2, name='gbn2', fix_gamma=True, eps=eps)
    gact2 = mx.sym.Activation(gbn2, name='gact2', act_type='relu')
    # 8x8
    gconv3 = mx.sym.Deconvolution(gact2, name='gconv3', kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                                  num_filter=ngf*2, no_bias=True)
    gbn3 = mx.sym.BatchNorm(gconv3, name='gbn3', fix_gamma=True, eps=eps)
    gact3 = mx.sym.Activation(gbn3, name='gact3', act_type='relu')
    # 16x16
    gconv4 = mx.sym.Deconvolution(gact3, name='gconv4', kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                                  num_filter=ngf, no_bias=True)
    gbn4 = mx.sym.BatchNorm(gconv4, name='gbn4', fix_gamma=True, eps=eps)
    gact4 = mx.sym.Activation(gbn4, name='gact4', act_type='relu')
    # 32x32
    gconv5 = mx.sym.Deconvolution(gact4, name='gconv5', kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                                  num_filter=nc, no_bias=True)
    gact5 = mx.sym.Activation(gconv5, name='gact5', act_type='tanh')
    G = gact5
    # discriminator
    data = mx.sym.Variable('data')
    # 64x64
    dconv1 = mx.sym.Convolution(data, name='dconv1', kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                                num_filter=ndf, no_bias=True)
    dact1 = mx.sym.LeakyReLU(dconv1, name='dact1', act_type='leaky', slope=0.2)
    # 32x32
    dconv2 = mx.sym.Convolution(dact1, name='dconv2', kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                                num_filter=ndf*2, no_bias=True)
    dbn2 = mx.sym.BatchNorm(dconv2, name='dbn2', fix_gamma=True, eps=eps)
    dact2 = mx.sym.LeakyReLU(dbn2, name='dact2', act_type='leaky', slope=0.2)
    # 16x16
    dconv3 = mx.sym.Convolution(dact2, name='dconv3', kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                                num_filter=ndf*4, no_bias=True)
    dbn3 = mx.sym.BatchNorm(dconv3, name='dbn3', fix_gamma=True, eps=eps)
    dact3 = mx.sym.LeakyReLU(dbn3, name='dact3', act_type='leaky', slope=0.2)
    # 8x8
    dconv4 = mx.sym.Convolution(dact3, name='dconv4', kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                                num_filter=ndf*8, no_bias=True)
    dbn4 = mx.sym.BatchNorm(dconv4, name='dbn4', fix_gamma=True, eps=eps)
    dact4 = mx.sym.LeakyReLU(dbn4, name='dact4', act_type='leaky', slope=0.2)
    # 4x4
    dconv5 = mx.sym.Convolution(dact4, name='dconv5', kernel=(4, 4), num_filter=1, no_bias=True)
    # 1x1
    D = mx.sym.Flatten(dconv5)
    return G, D


########## data part ##########

class RandIter(mx.io.DataIter):
    '''random data iterator for G
    '''
    def __init__(self, batch_size, ndim):
        super(RandIter, self).__init__()
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        return [mx.random.normal(0, 1, shape=(self.batch_size, self.ndim, 1, 1))]


class ImageIter(mx.io.DataIter):
    '''train data iterator for D
    '''
    def __init__(self, path, batch_size, data_shape):
        super(ImageIter, self).__init__()
        self.internal = mx.io.ImageRecordIter(
            path_imgrec=path,
            data_shape=data_shape,
            batch_size=batch_size,
            random_mirror=True)
        self.provide_data = [('data', (batch_size,) + data_shape)]
        self.provide_label = []

    def reset(self):
        self.internal.reset()

    def iter_next(self):
        return self.internal.iter_next()

    def getdata(self):
        data = self.internal.getdata()
        data = (data - 127.5) / 127.5
        return [data]


########## train part ##########

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


class WGANMetric(object):
    ''' metric for wgan
    '''

    def __init__(self):
        self.update_ = 0
        self.value_ = 0

    def reset(self):
        '''reset status
        '''
        self.update_ = 0
        self.value_ = 0

    def update(self, val):
        '''update metric
        '''
        self.update_ += 1
        self.value_ += val

    def get(self):
        '''get metric value
        '''
        return self.value_ / self.update_


def train():
    '''train wgan
    '''
    ctxs = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    batch_size = args.batch_size
    z_dim = args.z_dim
    lr = args.lr
    epoches = args.epoches
    wclip = args.wclip
    frequency = args.frequency
    model_prefix = args.model_prefix
    rand_iter = RandIter(batch_size, z_dim)
    image_iter = ImageIter(args.data_path, batch_size, (3, 64, 64))
    # G and D
    symG, symD = dcgan64x64(ngf=args.ngf, ndf=args.ndf, nc=args.nc)
    modG = mx.mod.Module(symbol=symG, data_names=('rand',), label_names=None, context=ctxs)
    modG.bind(data_shapes=rand_iter.provide_data)
    modG.init_params(initializer=mx.init.Normal(0.002))
    modG.init_optimizer(
        optimizer='sgd',
        optimizer_params={
            'learning_rate': lr,
        })
    modD = mx.mod.Module(symbol=symD, data_names=('data',), context=ctxs)
    modD.bind(data_shapes=image_iter.provide_data,
              inputs_need_grad=True)
    modD.init_params(mx.init.Normal(0.002))
    modD.init_optimizer(
        optimizer='sgd',
        optimizer_params={
            'learning_rate': lr,
        })
    # train
    logging.info('Start training')
    metricD = WGANMetric()
    metricG = WGANMetric()
    fix_noise_batch = mx.io.DataBatch([mx.random.normal(0, 1, shape=(batch_size, z_dim, 1, 1))], [])
    for epoch in range(epoches):
        image_iter.reset()
        metricD.reset()
        metricG.reset()
        for i, batch in enumerate(image_iter):
            # clip weight
            for params in modD._exec_group.param_arrays:
                for param in params:
                    mx.nd.clip(param, -wclip, wclip, out=param)
            # forward G
            rbatch = rand_iter.next()
            modG.forward(rbatch, is_train=True)
            outG = modG.get_outputs()
            # fake
            modD.forward(mx.io.DataBatch(outG, label=[]), is_train=True)
            fw_g = modD.get_outputs()[0].asnumpy()
            modD.backward([mx.nd.ones((batch_size, 1)) / batch_size])
            gradD = [[grad.copyto(grad.context) for grad in grads] for grads in modD._exec_group.grad_arrays]
            # real
            modD.forward(batch, is_train=True)
            fw_r = modD.get_outputs()[0].asnumpy()
            modD.backward([-mx.nd.ones((batch_size, 1)) / batch_size])
            for grads_real, grads_fake in zip(modD._exec_group.grad_arrays, gradD):
                for grad_real, grad_fake in zip(grads_real, grads_fake):
                    grad_real += grad_fake
            modD.update()
            errorD = -(fw_r - fw_g) / batch_size
            metricD.update(errorD.mean())
            # update G
            rbatch = rand_iter.next()
            modG.forward(rbatch, is_train=True)
            outG = modG.get_outputs()
            modD.forward(mx.io.DataBatch(outG, []), is_train=True)
            errorG = -modD.get_outputs()[0] / batch_size
            modD.backward([-mx.nd.ones((batch_size, 1)) / batch_size])
            modG.backward(modD.get_input_grads())
            modG.update()
            metricG.update(errorG.asnumpy().mean())
            # logging state
            if (i+1)%frequency == 0:
                print("epoch:", epoch+1, "iter:", i+1, "G: ", metricG.get(), "D: ", metricD.get())
        # save checkpoint
        modG.save_checkpoint('model/%s-G'%(model_prefix), epoch+1)
        modD.save_checkpoint('model/%s-D'%(model_prefix), epoch+1)
        rbatch = rand_iter.next()
        modG.forward(rbatch)
        outG = modG.get_outputs()[0]
        visual('tmp/gout-rand-%d.png'%(epoch+1), outG.asnumpy())
        modG.forward(fix_noise_batch)
        outG = modG.get_outputs()[0]
        visual('tmp/gout-fix-%d.png'%(epoch+1), outG.asnumpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data/face.rec', help="training data path")
    parser.add_argument('--batch-size', type=int, default=64, help="batch size")
    parser.add_argument('--wclip', type=float, default=0.01, help="weight clip for D")
    parser.add_argument('--gpus', type=str, default='0', help="gpu device id")
    parser.add_argument('--z-dim', type=int, default=100, help="z dimension")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--epoches', type=int, default=100, help="epoches to train")
    parser.add_argument('--frequency', type=int, default=100, help="frequence to logging traing status")
    parser.add_argument('--model-prefix', type=str, default='face', help="saved model prefix path")
    parser.add_argument('--ngf', type=int, default=64, help="base filter number of generator")
    parser.add_argument('--ndf', type=int, default=64, help="base filter number of discriminator")
    parser.add_argument('--nc', type=int, default=3, help="generator output channels")
    args = parser.parse_args()
    print(args)
    train()
