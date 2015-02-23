#!/usr/bin/env python3

import numpy as np
import theano
import theano.tensor as T

floatX = theano.config.floatX

from lbtoolbox import nnet_layers

class StreamingMLPSoftmax(object):

    def __init__(self, nfeat, batchsize, npred,
                 nhid=[1200], activ=[nnet_layers.Tanh],
                 l1pen=None, l2pen=None):
        self.batchsize = batchsize

        # Allocate symbolic variables for the data, and `strict` will make
        # sure to make transfers obvious!
        # TODO: is there a way to avoid the numpy allocation?
        self.sh_x = theano.shared(np.empty((batchsize, nfeat), dtype=floatX), 'x', strict=True)
        self.sh_y = theano.shared(np.empty((batchsize,), dtype=np.int32), 'y', strict=True)

        self.Ws = []
        self.bs = []

        hl = lambda: 1
        hl.out = self.sh_x
        hl.outshape = (batchsize, nfeat)
        for n, a in zip(nhid, activ):
            fc = nnet_layers.FullyConnected(hl.out, inshape=hl.outshape, outunits=n)
            self.Ws += fc.Ws
            self.bs += fc.bs
            hl = a(fc.out, fc.outshape)
            hl.init_incoming(fc.W, fc.b, fc.W_shape, fc.b_shape)
            self.Ws += hl.Ws
            self.bs += hl.bs

        self.softmax = nnet_layers.Softmax(hl.out, n_in=hl.outshape[1], n_classes=npred)
        self.Ws += self.softmax.Ws
        self.bs += self.softmax.bs

        self.params = self.Ws + self.bs

        # Symbolic theano expression to compute the negative log-likelihood
        self.nll = self.softmax.negLogLik(self.sh_y)

        # The cost only differs from the negative log-likelihood if there's regularization.
        self.cost = self.softmax.negLogLik(self.sh_y)

        # L1 norm; one regularization option is to enforce L1 norm to be small.
        if l1pen is not None:
            self.cost += l1pen * sum(T.sum(abs(W)) for W in self.Ws)

        # square L2 norm; one regularization option is to enforce L2 norm to be small.
        if l2pen is not None:
            #self.cost += l2pen * sum(T.sqrt(T.sum(W**2)) for W in self.Ws)
            self.cost += l2pen * sum(T.sum(W**2) for W in self.Ws)

        self.errors = self.softmax.nerrors(self.sh_y)

        # Symbolic theano expression for p(y|x)
        self.pred_proba = self.softmax.p_y_given_x


class StreamingLeNetSoftmax(object):
    def __init__(self, imshape, batchsize, npred,
                 nconv=[20, 50],
                 convshapes=[(5,5), (5,5)],
                 convstrides=[(1,1), (1,1)],
                 poolsizes=[(2,2), (2,2)],
                 poolstrides=[None, None],
                 nhid=[1200],
                 activ=[nnet_layers.Tanh, nnet_layers.Tanh, nnet_layers.Tanh],
                 l1pen=None, l2pen=None):
        self.batchsize = batchsize

        # If the imshape doesn't contain the depth, don't assume anything but fail.
        assert(len(imshape) == 3)
        assert(len(nconv) == len(convshapes) == len(convstrides) == len(poolsizes))

        # Allocate symbolic variables for the data, and `strict` will make
        # sure to make transfers obvious!
        # TODO: is there a way to avoid the numpy allocation?
        self.sh_x = theano.shared(np.empty((batchsize, np.product(imshape)), dtype=floatX), 'x', strict=True)
        self.sh_y = theano.shared(np.empty((batchsize,), dtype=np.int32), 'y', strict=True)

        self.Ws = []
        self.bs = []

        # Stack the convolution layers.
        l = lambda: 1
        l.outshape = (batchsize,) + imshape
        l.out = self.sh_x.reshape(l.outshape)
        for n, cs, ct, ps, pt, a in zip(nconv, convshapes, convstrides, poolsizes, poolstrides, activ):
            print("conv.in", l.outshape)
            conv = nnet_layers.Conv(l.out, n_im=l.outshape[0], imdepth=l.outshape[1], imshape=l.outshape[2:], n_conv=n, convshape=cs, stride=ct)
            print("conv.out", conv.outshape)
            maxp = nnet_layers.MaxPool(conv.out, ps, pt, inshape=conv.outshape)
            print("maxp.out", maxp.outshape)
            l = a(maxp.out, maxp.outshape)
            print("tanh.out", l.outshape)
            l.init_incoming(conv.W, conv.b, conv.W_shape, conv.b_shape, conv.fan_in, conv.fan_out)
            self.Ws += conv.Ws
            self.bs += conv.bs

        # Stack the fully-connected layers.
        for n, a in zip(nhid, activ[len(nconv):]):
            print("hl.in:", l.outshape)
            fc = nnet_layers.FullyConnected(l.out, inshape=l.outshape, outunits=n)
            print("hl.out:", fc.outshape)
            l = a(fc.out, fc.outshape)
            l.init_incoming(fc.W, fc.b, fc.W_shape, fc.b_shape)
            self.Ws += fc.Ws
            self.bs += fc.bs

        print("sm.in:", l.outshape)
        print("sm.out:", npred)
        self.softmax = nnet_layers.Softmax(l.out, n_in=l.outshape[1], n_classes=npred)
        self.Ws.append(self.softmax.W)
        self.bs.append(self.softmax.b)

        self.params = self.Ws + self.bs

        # Symbolic theano expression to compute the negative log-likelihood
        self.nll = self.softmax.negLogLik(self.sh_y)

        # The cost only differs from the negative log-likelihood if there's regularization.
        self.cost = self.softmax.negLogLik(self.sh_y)

        # L1 norm; one regularization option is to enforce L1 norm to be small.
        if l1pen is not None:
            self.cost += l1pen * sum(T.sum(abs(W)) for W in self.Ws)

        # square L2 norm; one regularization option is to enforce L2 norm to be small.
        if l2pen is not None:
            #self.cost += l2pen * sum(T.sqrt(T.sum(W**2)) for W in self.Ws)
            self.cost += l2pen * sum(T.sum(W**2) for W in self.Ws)

        self.errors = self.softmax.nerrors(self.sh_y)

        # Symbolic theano expression for p(y|x)
        self.pred_proba = self.softmax.p_y_given_x
