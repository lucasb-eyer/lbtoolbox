#!/usr/bin/env python3

import numpy as np
import theano
import theano.tensor as T

floatX = theano.config.floatX

# Quite a few things in here originate from the deeplearning tutorial on theano
# located at http://deeplearning.net/tutorial/

# Also still to test:
# - Ultra fast sigmoid: http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#tensor.nnet.ultra_fast_sigmoid
# - cuDNN, GpuCorrMM: http://deeplearning.net/software/theano/library/tensor/nnet/conv.html

def _weightparam(p, default, shape, name=None):
    if p is None:
        p = default(shape)
    if isinstance(p, np.ndarray):
        p = theano.shared(p.astype(floatX), name='{}{}'.format(name, shape))
    return p


class LinRegLayer(object):
    """
    Multi-objective linear-regression layer.
    """

    def __init__(self, X, n_in, n_preds=1, bias=True, w0=None, b0=None):
        """
        Creates the theano expression `self.y_pred` which corresponds to a
        linear regression.

        :type X: theano.tensor.TensorType
        :param X: Symbolic variable that is the input of the linreg layer, i.e.
                  a minibatch.

        :type n_in: int
        :param n_in: Number of input units, i.e. dimension of the space in
                     which the datapoints in `X` lie.

        :type n_preds: int
        :param n_preds: Number of values to be predicted.
        """

        # Initialize to all zeros for now.
        # TODO: Maybe add other initial values as optional parameter.
        self.W = _weightparam(w0, np.zeros, (n_in, n_preds), name='W_linreg')
        self.params = [self.W]

        # Computes the prediction value
        self.y_pred = T.dot(X, self.W)

        if bias:
            self.b = _weightparam(b0, np.zeros, (n_preds,), name='b_linreg')
            self.params += [self.b]
            self.y_pred += self.b.dimshuffle('x', 0)


    def rmse(self, y):
        """
        Returns a theano expression which computes the RMSE in the minibatch
        which is currently stored in whatever was given as `X` during init,
        over the total number of examples in it.

        RMSE means Root-Mean-Square-Error.

        :type y: theano.tensor.TensorType
        :param y: The correct label for each example.
        """
        return T.sqrt(T.mean((self.y_pred - y)**2))


    def mad(self, y):
        """
        Returns a theano expression which computes the MAD in the minibatch
        which is currently stored in whatever was given as `X` during init.

        MAD means Mean-Average-Deviation.

        :type y: theano.tensor.TensorType
        :param y: The correct label for each example.
        """
        return T.mean(abs(self.y_pred - y))


class SoftmaxLayer(object):
    """
    Softmax is another name for multi-class logistic-regression. This is the
    layer which implements it, though only with integer classes for now.
    """

    def __init__(self, X, n_in, n_classes, bias=True, w0=None, b0=None):
        """
        Creates two theano expressions:
        - `self.y_pred`: Computes the label of each datapoint in `X`.
        - `self.p_y_given_x`: Computes the probability of each of the
                              `n_classes` labels for each datapoint in `X`.

        :type X: theano.tensor.TensorType
        :param X: Symbolic variable that is the input of the layer, i.e. a minibatch.

        :type n_in: int
        :param n_in: Number of input units, i.e. dimension of the space in
                     which the datapoints in `X` lie.

        :type n_classes: int
        :param n_classes: Number of classes to be predicted.

        :type bias: bool
        :param bias: Whether to use additive bias or not.
        """

        self.W = _weightparam(w0, np.zeros, (n_in, n_classes), name='W_logreg')
        self.params = [self.W]
        lin = T.dot(X, self.W)
        if bias:
            self.b = _weightparam(b0, np.zeros, (n_classes,), name='b_logreg')
            self.params += [self.b]
            lin += self.b

        self.p_y_given_x = T.nnet.softmax(lin)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)


    def negLogLik(self, y):
        """
        Returns a theano expression which computes the mean negative
        log-likelihood (-log p(y|x,W)) of the minibatch which is currently
        stored in whatever was given as `X` during init.

        :type y: theano.tensor.TensorType
        :param y: The correct label for each example.
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch

        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1]

        # T.log(self.p_y_given_x) is a matrix of Log-Probabilities (call it LP)
        # with one row per example and one column per class

        # LP[T.arange(y.shape[0]),y] is a vector v containing [LP[0,y[0]],
        # LP[1,y[1]], LP[2,y[2]], ..., LP[n-1,y[n-1]]] and

        # T.mean(LP[T.arange(y.shape[0]),y]) is the mean (across minibatch
        # examples) of the elements in v, i.e., the mean log-likelihood across
        # the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


    def errors(self, y):
        """
        Returns a theano expression which computes the percentage of wrongly
        classified samples in the minibatch currently stored in `X`. This is
        also called the zero-one loss over the minibatch, or 1-score.

        :type y: theano.tensore.TensorType
        :param y: The correct label for each example.
        """
        return T.mean(T.neq(self.y_pred, y))

    def nerrors(self, y):
        """
        Returns a theano expression which computes the amount of wrongly
        classified samples in the minibatch currently stored in `X`.

        :type y: theano.tensore.TensorType
        :param y: The correct label for each example.
        """
        return T.sum(T.neq(self.y_pred, y))


initializations = {
    None: {
        'W': lambda shape: np.random.standard_normal(shape).astype(floatX),
        'b': lambda shape: np.zeros(shape, dtype=floatX),
    },
    T.tanh: {
        'W': lambda shape: np.random.uniform(
            low =-np.sqrt(6 / sum(shape)),
            high= np.sqrt(6 / sum(shape)),
            size=shape).astype(floatX),
        'b': lambda shape: np.zeros(shape, dtype=floatX)
    },
    T.nnet.sigmoid: {
        'W': lambda shape: np.random.uniform(
            low =-4*np.sqrt(6 / sum(shape)),
            high= 4*np.sqrt(6 / sum(shape)),
            size=shape).astype(floatX),
        'b': lambda shape: np.zeros(shape, dtype=floatX)
    }
}


class Nonlinearity(object):
    """
    Guess what?
    """

    def __init__(self, X, fn=T.tanh):
        self.X = X
        self.out = fn(self.X)
        self.params = []


# TODO: Split out a Nonlinearity class?
class HiddenLayer(object):
    """
    Typical hidden layer of a MLP: fully-connected with a non-linearity.
    """

    def __init__(self, X, n_in, n_out, activ=T.tanh, w0=None, b0=None, bias=True):
        """
        Creates a theano expression `self.out` which computes the linear
        transformation of the input by the weights followed by the nonlinearity:

        out = activ(X*W+b)

        Optionally, `activ` can be `None`.

        The default weight and bias initializations are taken from the global
        `initializations` dict, but can be overwritten in `w0`, `b0`.

        :type X: theano.tensor.TensorType
        :param X: Symbolic variable that is the input of the layer, i.e. a minibatch.

        :type n_in: int
        :param n_in: Number of input units, i.e. dimension of the space in
                     which the datapoints in `X` lie.

        :type n_out: int
        :param n_out: Number of outputs of this layer.

        :type activ: symbolic function
        :param activ: The nonlinearity function applied elementwise.

        :type w0: numpy.array
        :param w0: Optional non-default initialization for the weights. Should
                   have shape (n_in, n_out).

        :type b0: numpy.array
        :param b0: Optional non-default initialization for the biases. Should
                   have shape (n_out,).

        :type bias: bool
        :param bias: Whether to use additive bias or not.
        """
        self.X = X

        self.W = _weightparam(w0, initializations[activ]['W'], (n_in, n_out), 'W')
        self.params = [self.W]
        self.out = T.dot(X, self.W)

        if bias:
            self.b = _weightparam(b0, initializations[activ]['b'], (n_out,), 'b')
            self.params += [self.b]
            self.out += self.b

        if activ:
            self.out = activ(self.out)


# TODO
class DropoutLayer(object):
    pass


class ConvLayer(object):
    """
    Your neighborhood convolutional layer.
    """

    def __init__(self, X, n_im=None, imdepth=None, imshape=None, n_conv=None, convshape=None, stride=(1,1), w0=None, b0=None, bias=True):
        """
        Creates a theano expression `self.out` which computes the convolutions
        of the input (a minibatch of multi-channel images: 4D):

        out = conv2d(in) + b

        The output will be of shape...
        TODO

        The default weight and bias initializations are taken from the global
        `initializations` dict, but can be overwritten in `w0`, `b0`.

        :type X: theano.tensor.TensorType
        :param X: Symbolic variable that is the input of the layer, i.e. a minibatch.

        :type n_in: int
        :param n_in: Number of input units, i.e. dimension of the space in
                     which the datapoints in `X` lie.

        :type n_out: int
        :param n_out: Number of outputs of this layer.

        :type activ: symbolic function
        :param activ: The nonlinearity function applied elementwise.

        :type w0: numpy.array
        :param w0: Optional non-default initialization for the weights. Should
                   have shape (n_in, n_out).

        :type b0: numpy.array
        :param b0: Optional non-default initialization for the biases. Should
                   have shape (n_out,).

        :type bias: bool
        :param bias: Whether to use additive bias or not.
        """
        assert len(imshape) == 2, "imshape is supposed to be the image width and height, not " + str(imshape)
        assert len(convshape) == 2, "convshape is supposed to be the image width and height, not " + str(convshape)

        self.X = X

        # TODO: verify when on coffee. Done, was very wrong.
        # TODO: Take stride into account!
        self.outshape = tuple(i - c + 1 for i,c in zip(imshape, convshape))

        # Only used for initialization.
        fan_in = imdepth * np.prod(imshape)
        fan_out = n_conv * np.prod(self.outshape)  # TODO: verify when on coffee
                                                   # DONE: was stupid bullshit: n_conv * np.prod(convshape) <-- LOL
                                                   # BUT BUT: that's how it's done in theano?

        # Taken from the Theano deeplearning tutorial.
        W_shape = (n_conv, imdepth) + convshape
        W_bound = np.sqrt(6 / (fan_in + fan_out))

        if w0 is None:
            w0 = np.random.uniform(low=-W_bound, high=W_bound, size=W_shape)
        if isinstance(w0, np.ndarray):
            w0 = theano.shared(w0.astype(floatX), name='W_conv {}x{}'.format(n_conv, (imdepth,) + convshape))
        self.W = w0
        self.params = [self.W]

        self.out = T.nnet.conv.conv2d(self.X, self.W,
                image_shape=(n_im, imdepth) + imshape,
                filter_shape=W_shape,
                border_mode='valid', subsample=stride)

        if bias:
            if b0 is None:
                b0 = np.zeros(n_conv)
            if isinstance(b0, np.ndarray):
                b0 = theano.shared(b0.astype(floatX), name='b_conv {}'.format(n_conv))
            self.b = b0
            self.params += [self.b]
            self.out += self.b.dimshuffle('x', 0, 'x', 'x')


class MaxPoolingLayer(object):
    """
    Simple 2D max-pooling.
    """

    def __init__(self, X, poolsize, ignore_border=False, inshape=None):
        self.X = X
        self.out = T.signal.downsample.max_pool_2d(self.X, ds=poolsize, ignore_border=ignore_border)
        self.params = []

        if inshape:
            if ignore_border:
                self.outshape = tuple(s // p for s, p in zip(inshape, poolsize))
            else:
                # (s+p-1) // p  is the same, but more obfuscated/neat!
                def sz(s, p):
                    q, r = divmod(s, p)
                    return q if r == 0 else q+1
                self.outshape = tuple(map(sz, inshape, poolsize))
