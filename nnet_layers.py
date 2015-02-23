#!/usr/bin/env python3

import numpy as np
import theano
import theano.tensor as T

floatX = theano.config.floatX

# TODO: Formalize!
# A Layer should always have:
# - outshape: shape of the output, *with* leading minibatch size.
# - Ws: a list of weight parameters.
# - bs: a list of bias parameters.
# - params: a list of all parameters.

# Also somewhat standarized:
# - W_shape
# - b_shape
# - fan_in
# - fan_out

# Still to test and maybe add:
# - Ultra fast sigmoid: http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#tensor.nnet.ultra_fast_sigmoid
# - cuDNN, GpuCorrMM: http://deeplearning.net/software/theano/library/tensor/nnet/conv.html


def _param(p, shape, name, default):
    if p is None:
        p = default(shape)
    if isinstance(p, np.ndarray):
        p = theano.shared(p.astype(floatX).reshape(shape), name='{}{}'.format(name, shape))
    return p


class LinReg(object):
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

        self.W_shape = (n_in, n_preds)
        self.W = _param(w0, self.W_shape, 'W_linreg', np.zeros)

        # Computes the prediction value
        self.y_pred = T.dot(X, self.W)

        self.b_shape = (n_preds,)
        if bias:
            self.b = _param(b0, self.b_shape, 'b_linreg', np.zeros)
            self.y_pred += self.b.dimshuffle('x', 0)

        self.Ws = [self.W]
        self.bs = [self.b] if bias else []
        self.params = self.Ws + self.bs


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


class Softmax(object):
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

        self.W_shape = (n_in, n_classes)
        self.W = _param(w0, self.W_shape, 'W_logreg', np.zeros)

        lin = T.dot(X, self.W)

        if bias:
            self.b_shape = (n_classes,)
            self.b = _param(b0, self.b_shape, 'b_logreg', np.zeros)
            lin += self.b

        self.Ws = [self.W]
        self.bs = [self.b] if bias else []
        self.params = self.Ws + self.bs

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

        # TODO: Wait for https://github.com/Theano/Theano/issues/2464
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


    def errors(self, y):
        """
        Returns a theano expression which computes the percentage of wrongly
        classified samples in the minibatch currently stored in `X`. This is
        also called the zero-one loss over the minibatch, or 1-score.

        :type y: theano.tensore.TensorType
        :param y: The correct label for each example.
        """
        return T.mean(T.neq(self.y_pred[:y.shape[0]], y))

    def nerrors(self, y):
        """
        Returns a theano expression which computes the amount of wrongly
        classified samples in the minibatch currently stored in `X`.

        :type y: theano.tensore.TensorType
        :param y: The correct label for each example.
        """
        return T.sum(T.neq(self.y_pred[:y.shape[0]], y))


class Tanh(object):
    """
    Typical tanh nonlinearity layer of oldschool nnets.
    """

    def __init__(self, X, inshape=None, sigmoid=False):
        """
        Creates a theano expression `self.out` which computes the elementwise
        application of the tanh function:

        out = tanh(X)

        :type X: theano.tensor.TensorType
        :param X: Symbolic variable that is the input of the layer, i.e. a minibatch.

        :type inshape: tuple
        :param inshape: Size of the input. Only used for setting `outshape` to it.

        :type sigmoid: boolean
        :param sigmoid: Whether to use the sigmoidal instead of tanh.
        """
        self.sigmoid = sigmoid
        self.X = X
        self.Ws = self.bs = self.params = []
        self.out = T.tanh(X) if not sigmoid else T.nnet.sigmoid(X)
        self.outshape = inshape


    def init_incoming(self, W, b, W_shape, b_shape=None, fan_in=None, fan_out=None):
        # "Xavier" initialization:
        # Understanding the difficulty of training deep feedforward neural networks
        fan_in = fan_in or W_shape[0]
        fan_out = fan_out or W_shape[1]
        b_shape = b_shape or (fan_out,)
        s = 1 if not self.sigmoid else 4
        W.set_value(np.random.uniform(
            low =-s*np.sqrt(6 / (fan_in+fan_out)),
            high= s*np.sqrt(6 / (fan_in+fan_out)),
            size=W_shape).astype(W.dtype))
        b.set_value(np.zeros(b_shape, dtype=b.dtype))


class ReLU(object):
    """
    Typical ReLU nonlinearity layer of oldschool nnets.
    """

    def __init__(self, X, inshape=None, leak=0, cap=None):
        """
        Creates a theano expression `self.out` which computes the elementwise
        application of the ReLU or leaky ReLU function:

        out = max(leak*X, X)

        `X`: Symbolic variable that is the input of the layer, i.e. a minibatch.

        `leak`: Number (TODO or symbolic variable) to use as slope of the "0"
                part. Defaults to 0, i.e. standard ReLU.

        `cap`: Number (TODO or symbolic variable) to use as max value of the
               linear part, as in Alex' CIFAR Convolutional Deep Belief Nets.
               If `None` (the default), do not apply any capping.

        `inshape`: Size of the input. Only used for setting `outshape` to it.
        """
        self.X = X
        # TODO: if `leak` is a Theano variable, add it to parameters.
        self.Ws = self.bs = self.params = []
        self.out = T.maximum(leak*X, X)
        if cap is not None:
            self.out = T.maximum(self.out, cap)
        self.outshape = inshape


    def init_incoming(self, W, b, W_shape, b_shape=None, fan_in=None, fan_out=None):
        # From "Delving Deep into Rectifiers"
        # but using the mean of fan_in and fan_out, i.e. (fi+fo)/2
        fan_in = fan_in or W_shape[0]
        fan_out = fan_out or W_shape[1]
        b_shape = b_shape or (fan_out,)
        var = np.sqrt(4/(fan_in+fan_out))
        W.set_value(var*np.random.standard_normal(W_shape).astype(W.dtype))
        b.set_value(np.zeros(b_shape, dtype=b.dtype))


class FullyConnected(object):
    """
    Fully-connected layer is the typical "hidden" layer. Implements a GEMV.
    """

    def __init__(self, X, inshape, outunits, w0=None, b0=None, bias=True):
        """
        Creates a theano expression `self.out` which computes the linear
        transformation of the input by the weights:

        out = X*W+b

        The default weight and bias initializations are uniform random. This is
        usually a bad idea, thus it's recommended to let the following
        nonlinearity layer initialize the weights.

        Alternatively, initial values can be given in `w0`, `b0`.

        :type X: theano.tensor.TensorType
        :param X: Symbolic variable that is the input of the layer, i.e. a minibatch.

        :type inshape: int or tuple of ints
        :param inshape: Number of input units, i.e. dimension of the space in
                        which the datapoints in `X` lie. If a tuple, the input
                        will be flattened. Including batchsize.

        :type outunits: int or tuple of ints
        :param outunits: Number of outputs of this layer. If a tuple, the
                     output will be reshaped to this. NOT including batchsize.

        :type w0: numpy.array or theano tensor
        :param w0: Optional non-default initialization for the weights. Should
                   have shape (prod(inshape), prod(outshape)).

        :type b0: numpy.array or theano tensor
        :param b0: Optional non-default initialization for the biases. Should
                   have shape (prod(outshape),).

        :type bias: bool
        :param bias: Whether to use additive bias `b` or not.
        """
        # Tupleize:
        outunits = outunits if isinstance(outunits, tuple) else (outunits,)

        self.X = X

        self.fan_in = np.prod(inshape[1:])
        self.fan_out = np.prod(outunits)

        self.W_shape = (self.fan_in, self.fan_out)
        self.W = _param(w0, self.W_shape, 'W_fc', np.random.standard_normal)

        # For non-1D inputs, add a flattening step.
        if isinstance(inshape, tuple) and len(inshape) > 2:
            # Flatten all dimensions, but keep the batchsize one.
            X = X.flatten(2)

        self.out = T.dot(X, self.W)

        # For non-1D outputs, add the reshaping step.
        if len(outunits) > 1:
            self.out = self.out.reshape((inshape,) + outunits)

        self.outshape = (inshape[0],) + outunits

        if bias:
            self.b_shape = outunits
            self.b = _param(b0, self.b_shape, 'b_fc', np.zeros)
            self.out += self.b

        self.Ws = [self.W]
        self.bs = [self.b] if bias else []
        self.params = self.Ws + self.bs


# TODO
class DropoutLayer(object):
    pass


class Conv(object):
    """
    Your local neighborhood convolutional layer.
    """

    def __init__(self, X, n_im, imdepth, imshape, n_conv,
                 convshape, stride=(1,1), border_mode='valid',
                 w0=None, b0=None, bias=True):
        """
        Creates a theano expression `self.out` which computes the convolutions
        of the input (a minibatch of multi-channel images: 4D):

        out = conv2d(in) + b

        More specifically, it computes each output feature map out_i as:

        out_i = ... TODO

        The default weight and bias initializations are uniform random. This is
        usually a bad idea, thus it's recommended to let the following
        nonlinearity layer initialize the weights.

        Alternatively, initial values can be given in `w0`, `b0`.

        - TODO: Take stride into account for size computations!
        - TODO: Support 'full' mode for size computations!

        `X`: Symbolic theano variable that is the 4-dimensional input of the
             layer, shaped as (#batches, #channels, height, width)

        `n_im`: Number of input images. Usually the size of a mini-batch.

        `imdepth`: Number of input feature-maps per image. E.g. 3 for RGB.

        `imshape`: The shape (h,w) of each image.

        `n_conv`: Number of output feature-maps per image, i.e. number of filters.

        `convshape`: The shape (h,w) of the filters.

        `stride`: The stride of the convolution operation; defaults to (1,1).
                  In Theano, this is implemented as subsampling the output, so
                  it does not increase the speed of the convolution!

        `border_mode`: From Theano's documentation:
            - "valid": only apply filter to complete patches of the image.
                Output shape: image_shape - filter_shape + 1
            - "full": zero-pads image to multiple of filter shape.
                Output shape: image_shape + filter_shape - 1

        `w0`: Optional numpy array or theano tensor used to initialize `W`.
            If a theano tensor, will become shared weights.
            Should have shape (n_conv, imdepth, *convshape).

        `b0`: Optional numpy array or theano tensor used to initialize `b`.
            If a theano tensor, will become shared weights.
            Should have shape (n_out,).

        `bias`: Whether to use additive bias or not.
        """
        assert len(imshape) == 2, "imshape is supposed to be the image width and height, not " + str(imshape)
        assert len(convshape) == 2, "convshape is supposed to be the image width and height, not " + str(convshape)

        self.X = X

        # TODO: verify when on coffee. Done, was very wrong.
        # TODO: Take stride into account!
        # TODO: Support 'full' mode!
        self.outshape = (n_im, n_conv,) + tuple(i - c + 1 for i,c in zip(imshape, convshape))

        # Only used for initialization.
        self.fan_in = imdepth * np.prod(imshape)
        # TODO: verify when on coffee
        # DONE: was stupid bullshit: n_conv * np.prod(convshape) <-- LOL
        # BUT BUT: that's how it's done in theano?
        self.fan_out = n_conv * np.prod(self.outshape[1:])

        self.W_shape = (n_conv, imdepth) + convshape
        self.b_shape = (n_conv,)

        # Taken from the Theano deeplearning tutorial.
        def initW(shape):
            bound = np.sqrt(6 / (self.fan_in + self.fan_out))
            return np.random.uniform(low=-bound, high=bound, size=shape)
        self.W = _param(w0, self.W_shape, 'W_conv', initW)

        self.out = T.nnet.conv.conv2d(self.X, self.W,
                image_shape=(n_im, imdepth) + imshape,
                filter_shape=self.W_shape,
                border_mode=border_mode, subsample=stride)

        if bias:
            self.b = _param(b0, self.b_shape, 'b_conv', np.zeros)
            self.out += self.b.dimshuffle('x', 0, 'x', 'x')

        self.Ws = [self.W]
        self.bs = [self.b] if bias else []
        self.params = self.Ws + self.bs


class MaxPool(object):
    """
    Simple 2D max-pooling.
    """

    def __init__(self, X, poolsize, stride=None, ignore_border=False, inshape=None):
        """
        Creates a theano expression `self.out` which computes the max-pooling
        of the input over the last two dimensions.

        TODO: `stride` is waiting for https://github.com/Theano/Theano/issues/2196
        TODO: Implement shape computation using `stride`.

        `X`: Symbolic theano variable that is the N-dimensional input of the
             layer, shaped as (?, ..., height, width)

        `poolsize`: How large of a patch should be max-pooled (h,w)

        `stride`: The stride between pools, None means `poolsize`.

        `ignore_border`: Whether or not to take into account the border.

        `inshape`: Shape of the input, necessary for computing `outshape`.
        """
        self.X = X
        #self.out = T.signal.downsample.max_pool_2d(self.X, ds=poolsize, ignore_border=ignore_border, st=stride)
        self.out = T.signal.downsample.max_pool_2d(self.X, ds=poolsize, ignore_border=ignore_border)
        self.params = []

        if inshape:
            # "Max pooling will be done over the 2 last dimensions."
            self.outshape = inshape[:-2]
            if ignore_border:
                self.outshape += tuple(s // p for s, p in zip(inshape[-2:], poolsize))
            else:
                # (s+p-1) // p  is the same, but more obfuscated/neat!
                def sz(s, p):
                    q, r = divmod(s, p)
                    return q if r == 0 else q+1
                self.outshape += tuple(map(sz, inshape[-2:], poolsize))
