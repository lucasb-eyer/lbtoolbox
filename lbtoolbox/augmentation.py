#!/usr/bin/env python3
# coding: utf-8

import numpy as _np
import itertools as _it

# TODO: Implement in terms of skimage?
import scipy.ndimage.interpolation as _spint

import lbtoolbox.util as _u


class AugmentationPipeline(object):
    """
    A utility-class that keeps track of various augmenters, the image shape,
    and applies them to batches.
    """
    def __init__(self, Xtr, ytr, *augmenters):
        """
        - `Xtr` and `ytr` are the training dataset, as NxD and N arrays.

        - `augmenters` is a list of instances of implementations of `Augmenter`.
        """
        self.augmenters = list(augmenters)

        for a in self.augmenters:
            a.fit(Xtr, ytr)


    def _pred_indices(self, fast):
        return _it.product(*(range(aug.npreds(fast)) for aug in self.augmenters))


    def augimg_train(self, image, *targets):
        """
        Returns an "augmented" copy of the given `image`.
        This one is intendend for training, it's stochastic.

        `image` may be either 2D or flattened 1D.
        """

        img = image
        for a in self.augmenters:
            img, targets = a.transform_train(img, *targets)
        return img, targets


    def augimg_pred(self, image, fast=False):
        """
        Yields through "augmented" copies of the given `image`.
        This one is intended for testing, it will always yield the same sequence
        of augmentations.

        `image` may be either 2D or flattened 1D.
        """

        # Go through all possible combinations of transforms we get from the
        # augmenters for prediction.
        for iaug in self._pred_indices(fast):
            img = image
            for ia, a in zip(iaug, self.augmenters):
                img = a.transform_pred(img, ia, fast)
            yield img


    def augbatch_train(self, batch, *targets):
        """
        Returns an "augmented" copy of the given `batch` of images.
        This one is intendend for training, it's stochastic.

        `batch` has shape BxD where B is the number of samples in the batch and
        D is the dimensionality of a sample (784 or 28,28 or 1,28,28 for MNIST).
        """
        B = batch.shape[0]

        # Reserve space for the output
        out = None
        outtgts = None

        # Transform the images, one by one, independently.
        # Otherwise, the whole batch would be correlated.
        for i in range(B):
            img, tgts = self.augimg_train(batch[i], *(t[i] for t in targets))
            if out is None:
                out = _np.empty((B,) + img.shape, dtype=img.dtype)
                outtgts = tuple(_np.empty((B,) + t.shape, dtype=t.dtype) for t in tgts)
            out[i] = img
            for ot, t in zip(outtgts, tgts):
                ot[i] = t
        return (out,) + outtgts


    def augbatch_pred(self, batch, fast=False):
        """
        Yields through "augmented" copies of the given `batch` of images.
        This one is intended for testing, it will always yield the same sequence
        of augmentations.

        `batch` has shape BxD where B is the number of samples in the batch and
        D is the dimensionality of a sample (768 or 28,28 or 1,28,28 for MNIST).
        """
        B = batch.shape[0]

        # Reserve space for the output
        out = None

        # Go through all possible combinations of transforms we get from the
        # augmenters for prediction.
        for iaug in self._pred_indices(fast):
            # Transform the images, one by one, independently.
            # Otherwise, the whole batch would be correlated.
            for i in range(B):
                img = batch[i]
                for ia, a in zip(iaug, self.augmenters):
                    img = a.transform_pred(img, ia, fast)
                if out is None:
                    out = _np.empty((B,) + img.shape, dtype=img.dtype)
                out[i] = img
            yield out


class Augmenter(object):
    """
    The base-class for dataset augmenters.
    """


    def npreds(self, fast):
        """
        Return how many augmentations this augmenter generates at test-time.
        """
        return 1


    def fit(self, Xtr, ytr):
        """
        May be used to learn some dataset-specific statistics for augmentation.
        """
        pass


    def transform_train(self, img, *targets):
        """
        Randomly transform the given `img` for generating a new training sample.
        """
        return img, targets


    def transform_pred(self, img, i, fast=False):
        """
        Apply the `i`-th transformation of `img` for testing/prediction.
        """
        assert(i == 0)
        return img


class Flattener(Augmenter):
    """
    Simply flattens what comes in.
    """


    def transform_train(self, img, *targets):
        return img.flat, targets


    def transform_pred(self, img, *a, **kw):
        return img.flat


class Reshaper(Augmenter):
    """
    Simply reshapes what comes in.
    """


    def __init__(self, shape):
        self.shape = shape


    def transform_train(self, img, *targets):
        return img.reshape(self.shape), targets


    def transform_pred(self, img, *a, **kw):
        return img.reshape(self.shape)


class Flipper(Augmenter):
    """
    Flips the image across one or multiple dimensions.
    This is a generalization of horizontal/vertical flips.
    """


    def __init__(self, dims):
        """
        `dims` is a list or tuple of dimensions which should be flipped.
        """
        self.dims = dims


    def npreds(self, fast):
        return 2**len(self.dims)


    def transform_train(self, img, *targets):
        for d in self.dims:
            if _np.random.random() < 0.5:
                img = _u.flipany(img, d)
        return img, targets


    def transform_pred(self, img, i, fast):
        assert i < self.npreds(fast), "This should never happen, please file an issue."

        for idim, d in enumerate(self.dims):
            if i >> idim & 1:
                img = _u.flipany(img, d)
        return img


class Cropper(Augmenter):
    """
    A typical Krizhevsky-style random cropper.
    Generates random crops of a given size during training and generates
    five crops (4 corners + center) during prediction.

    TODO: At some point this should be fully generalized beyond 2D.
          It's partially there already by parametrizing the x,y dims.
    """


    def __init__(self, outshape, ydim=-2, xdim=-1):
        """
        Currently only for 2D images. `outshape` should contain two numbers,
        the first for height and the second for width.

        `ydim` and `xdim` are used to access the y and x dimensions in the
        data. The defaults of `-2` and `-1` work well in the default case.
        """
        self.osh = outshape
        self.ydim = ydim
        self.xdim = xdim

        assert len(outshape) == 2, "Currently, only cropping of 2D images is supported. Please file an issue if you need 3D."


    def npreds(self, fast):
        return 5 if not fast else 1


    def transform_train(self, img, *targets):
        dx = _np.random.randint(img.shape[self.xdim] - self.osh[1])
        dy = _np.random.randint(img.shape[self.ydim] - self.osh[0])

        slicing = [slice(None)] * len(img.shape)
        slicing[self.xdim] = slice(dx, dx+self.osh[1])
        slicing[self.ydim] = slice(dy, dy+self.osh[0])
        return img[slicing], targets


    def transform_pred(self, img, i, fast=False):
        if fast or i == 0:  # Center
            dx = (img.shape[self.xdim] - self.osh[1])//2
            dy = (img.shape[self.ydim] - self.osh[0])//2
            sx = slice(dx, dx+self.osh[1])
            sy = slice(dy, dy+self.osh[0])
        elif i == 1:  # Top-left
            sx = slice(None, self.osh[1])
            sy = slice(None, self.osh[0])
        elif i == 2:  # Top-right
            sx = slice(img.shape[self.xdim] - self.osh[1], None)
            sy = slice(None, self.osh[0])
        elif i == 3:  # Bottom-left
            sx = slice(None, self.osh[1])
            sy = slice(img.shape[self.ydim] - self.osh[0], None)
        elif i == 4:  # Bottom-right
            sx = slice(img.shape[self.xdim] - self.osh[1], None)
            sy = slice(img.shape[self.ydim] - self.osh[0], None)
        else:
            assert False, "This should never happen. Please file an issue."

        slicing = [slice(None)] * len(img.shape)
        slicing[self.xdim] = sx
        slicing[self.ydim] = sy
        return img[slicing]


class Rotator(Augmenter):
    """
    Augments an image by rotating it.

    TODO: Has not been tested with color-images yet!
    """


    def __init__(self, start=0, stop=180, npred=5, preds_fast=[0, 90], highqual=False):
        """
        Creates an augmenter which rotates the input image.
        If using together with flippers, keep in mind that:

        - 180° rotation == horiz + vert flips
        - 270° rotation == 90° rotation + horiz + vert flips

        Thus when using a horizontal flipper, only [0°, 180°] is necessary, and
        when using both horizontal and vertical flippers, only [0°, 90°] is.

        `start`: The smallest rotation angle to use.

        `stop`: The largest rotation angle to use.

        `npred`: How many orientations to use for prediction.

        `preds_fast`: List of orientations to use for "fast" prediction.

        `highqual`: Whether or not to use a higher-quality but roughly twice
                    slower interpolation algorithm.
        """
        self.pred_angles = {
            True: preds_fast,
            False: _np.linspace(start, stop, npred)
        }
        if highqual:
            self.order = 3
            self.prefilter = True
        else:
            self.order = 1
            self.prefilter = False


    def npreds(self, fast):
        return len(self.pred_angles[fast])


    def transform_train(self, img, *targets):
        deg = _np.random.uniform(self.pred_angles[False][0], self.pred_angles[False][-1])
        return _spint.rotate(img, deg,
            reshape=False, mode='nearest',
            order=self.order, prefilter=self.prefilter), targets


    def transform_pred(self, img, i, fast=False):
        return _spint.rotate(img, self.pred_angles[fast][i],
            reshape=False, mode='nearest',
            order=self.order, prefilter=self.prefilter)


class Zoomer(Augmenter):
    """
    Augments an image by zooming it.

    TODO: Has not been tested with color-images yet!
    """


    def __init__(self, range=(1/1.2, 1.2), highqual=False, cval=0):
        self.logrange = _np.log(range)
        self.kw = dict(order=3 if highqual else 1, mode='constant', cval=cval, prefilter=highqual)


    def npreds(self, fast):
        if fast:
            return 1
        else:
            return 3


    def transform_train(self, img, *targets):
        f0 = _np.exp(_np.random.uniform(*self.logrange))
        f1 = _np.exp(_np.random.uniform(*self.logrange))
        return self._zoomit(img, (f0, f1)), targets


    def transform_pred(self, img, i, fast=False):
        if fast or i == 0:
            return img
        else:
            f = _np.exp(self.logrange[i-1])
            return self._zoomit(img, (f,f))


    def _zoomit(self, img, factors):
        assert img.ndim == 2, "TODO: Currently not implemented for 3D images."

        zimg = _spint.zoom(img, factors, **self.kw)
        out = _np.full_like(img, self.kw['cval'])

        if zimg.shape[0] < out.shape[0]:
            dst_y0 = (out.shape[0] - zimg.shape[0])//2
            dst_y1 = dst_y0 + zimg.shape[0]
            src_y0 = 0
            src_y1 = zimg.shape[0]
        else:
            dst_y0 = 0
            dst_y1 = out.shape[0]
            src_y0 = (zimg.shape[0] - out.shape[0])//2
            src_y1 = src_y0 + out.shape[0]

        if zimg.shape[1] < out.shape[1]:
            dst_x0 = (out.shape[1] - zimg.shape[1])//2
            dst_x1 = dst_x0 + zimg.shape[1]
            src_x0 = 0
            src_x1 = zimg.shape[1]
        else:
            dst_x0 = 0
            dst_x1 = out.shape[1]
            src_x0 = (zimg.shape[1] - out.shape[1])//2
            src_x1 = src_x0 + out.shape[1]

        out[dst_y0:dst_y1,dst_x0:dst_x1] = zimg[src_y0:src_y1,src_x0:src_x1]
        return out


class ColorPCA(Augmenter):
    def __init__(self, std=0.1):
        self.std = std

    def fit(self, Xtr, ytr):
        # From (MiniBatch, Channel, ...) to (Channel, MiniBatch, ...)
        Xtr = _np.rollaxis(Xtr, 1)
        # To (Channel, MiniBatch*H*W)
        Xtr = Xtr.reshape((Xtr.shape[0], -1))
        self.l, self.V = _np.linalg.eig(_np.cov(Xtr))

    def transform_train(self, img, *targets):
        alpha = _np.random.randn(3)*self.std
        noise = _np.dot(self.V, alpha * self.l)
        # TODO: Find a better way to broadcast this addition!
        return img + noise[:,_np.newaxis,_np.newaxis], targets


class Gamma(Augmenter):
    """
    Augment images by randomly adjusting the gamma value.
    """
    def __init__(self, range=(0.25, 4.0)):
        self.logrange = _np.log(range)

    def npreds(self, fast):
        return 1 if fast else 3

    def transform_train(self, img, *targets):
        g = _np.exp(_u.truncrandn_approx(*self.logrange))
        return img**g, targets

    def transform_pred(self, img, i, fast=False):
        if fast or i == 0:
            return img
        else:
            return img**_np.exp(self.logrange[i-1])
