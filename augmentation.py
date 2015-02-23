#!/usr/bin/env python3

import numpy as _np
import itertools as _it


class AugmentationPipeline(object):
    """
    A utility-class that keeps track of various augmenters, the image shape,
    and applies them to batches.
    """
    def __init__(self, shape, Xtr, ytr, *augmenters):
        """
        `shape` is the actual shape of the image, e.g. (28,28) for MNIST.

        `Xtr` and `ytr` are the training dataset, as NxD and N arrays.

        `augmenters` is a list of instances of implementations of `Augmenter`.
        """
        self.imshape = shape
        self.augmenters = list(augmenters)

        for a in self.augmenters:
            a.fit(Xtr, ytr)


    def outshape(self, flat=False):
        """
        Compute the shape of the result of passing an image of `inshape`
        through my augmentations serially. Optionally, the shape is then
        `flat`tened.
        """
        sh = self.imshape
        for a in self.augmenters:
            sh = a.outshape(sh)
        return sh if not flat else _np.product(sh)


    def _pred_indices(self, fast):
        return _it.product(*(range(aug.npreds(fast)) for aug in self.augmenters))


    def aug_train(self, image):
        """
        Returns an "augmented" copy of the given `image`.
        This one is intendend for training, it's stochastic.

        `image` may be either 2D or flattened 1D.
        """
        img = image.reshape(self.imshape)

        for a in self.augmenters:
            img = a.transform_train(img)

        if len(image.shape) == 2:
            return img
        else:
            return img.flatten()


    def aug_pred(self, image, fast=False):
        """
        Yields through "augmented" copies of the given `image`.
        This one is intended for testing, it will always yield the same sequence
        of augmentations.

        `image` may be either 2D or flattened 1D.
        """
        # Go through all possible combinations of transforms we get from the
        # augmenters for prediction.
        for iaug in self._pred_indices(fast):
            img = image.reshape(self.imshape)
            for ia, a in zip(iaug, self.augmenters):
                img = a.transform_pred(img, ia)

            if len(image.shape) == 2:
                yield img
            else:
                yield img.flatten()


    def augbatch_train(self, images):
        """
        Returns an "augmented" copy of the given batch of `images`.
        This one is intendend for training, it's stochastic.

        `images` has shape BxD where B is the number of samples in the batch
        and D is the dimensionality of a sample (768 for MNIST).
        """
        B = images.shape[0]

        # Reserve space for the output
        out = _np.empty((B, self.outshape(flat=True)), dtype=images.dtype)

        # Transform the images, one by one, independently.
        # Otherwise, the whole batch would be correlated.
        for i in range(B):
            img = images[i].reshape(self.imshape)
            for a in self.augmenters:
                img = a.transform_train(img)
            out[i,:] = img.flat
        return out


    def augbatch_pred(self, images, fast=False):
        """
        Yields through "augmented" copies of the given batch of `images`.
        This one is intended for testing, it will always yield the same sequence
        of augmentations.

        `images` has shape BxD where B is the number of samples in the batch and
        D is the dimensionality of a sample (768 for MNIST).
        """
        B = images.shape[0]

        # Reserve space for the output
        out = _np.empty((B, self.outshape(flat=True)), dtype=images.dtype)

        # Go through all possible combinations of transforms we get from the
        # augmenters for prediction.
        for iaug in self._pred_indices(fast):
            # Transform the images, one by one, independently.
            # Otherwise, the whole batch would be correlated.
            for i in range(B):
                img = images[i].reshape(self.imshape)
                for ia, a in zip(iaug, self.augmenters):
                    img = a.transform_pred(img, ia)
                out[i,:] = img.flat
            yield out


    def precompile(self, images, fast, key):
        """
        The signature is the same as that of `augbatch_pred`.

        Precomputes all prediction augmentations on a batch of `images` and
        stores the result so it can be retrieved later on using the same `key`.
        """
        # Reserve space for the cache
        N = images.shape[0]
        out = _np.empty((N, self.outshape(flat=True)), dtype=images.dtype)

        # Go through all possible combinations of transforms we get from the
        # augmenters for prediction.
        for iaug in _it.product(*(range(aug.npreds(fast)) for aug in self.augmenters)):
            # Transform the images, one by one, independently.
            # Otherwise, the whole batch would be correlated.
            for i in range(B):
                img = images[i].reshape(self.imshape)
                for ia, a in zip(iaug, self.augmenters):
                    img = a.transform_pred(img, ia)
                out[i,:] = img.flat
            yield out


class Augmenter(object):
    """
    The base-class for dataset augmenters.
    """


    def npreds(self, fast=False):
        """
        Return how many augmentations this augmenter generates at test-time.
        """
        return 1


    def fit(self, Xtr, ytr):
        """
        May be used to learn some dataset-specific statistics for augmentation.
        """
        pass


    def outshape(self, inshape):
        """
        Return the shape of the transform of an image of `inshape` shape.
        """
        return inshape


    def transform_train(self, img):
        """
        Randomly transform the given `img` for generating a new training sample.
        """
        return img


    def transform_pred(self, img, i, fast=False):
        """
        Apply the `i`-th transformation of `img` for testing/prediction.
        """
        assert(i == 0)
        return img


class HorizontalFlipper(Augmenter):
    """
    Just flip the image horizontally.
    """
    def npreds(self, fast=False):
        return 2

    def transform_train(self, img):
        return _np.fliplr(img) if _np.random.random() < 0.5 else img

    def transform_pred(self, img, i, fast=False):
        if i == 0:
            return img
        elif i == 1:
            return _np.fliplr(img)
        else:
            assert(False)


class Cropper(Augmenter):
    """
    A typical Krizhevsky-style random cropper.
    Generates random crops of a given size during training and generates
    five crops (4 corners + center) during prediction.
    """


    def __init__(self, outshape):
        self.osh = outshape


    def npreds(self, fast=False):
        return 5 if not fast else 1


    def outshape(self, inshape):
        assert(self.osh[0] < inshape[0])
        assert(self.osh[1] < inshape[1])
        return self.osh


    def transform_train(self, img):
        dx = _np.random.randint(img.shape[1] - self.osh[1])
        dy = _np.random.randint(img.shape[0] - self.osh[0])
        return img[dy:dy+self.osh[0], dx:dx+self.osh[1]]


    def transform_pred(self, img, i, fast=False):
        if fast or i == 0:  # Center
            dx = (img.shape[1] - self.osh[1])//2
            dy = (img.shape[0] - self.osh[0])//2
            return img[dy:dy+self.osh[0], dx:dx+self.osh[1]]
        elif i == 1:  # Top-left
            return img[:self.osh[0], :self.osh[1]]
        elif i == 2:  # Top-right
            return img[:self.osh[0], img.shape[1]-self.osh[1]:]
        elif i == 3:  # Bottom-left
            return img[img.shape[0]-self.osh[0]:, :self.osh[1]]
        elif i == 4:  # Bottom-right
            return img[img.shape[0]-self.osh[0]:, img.shape[1]-self.osh[1]:]
        else:
            assert(False)


# TODO: Implement in terms of skimage too?
import scipy.ndimage.interpolation as _spint


class Rotator(Augmenter):
    """
    Augments an image by rotating it.
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


    def npreds(self, fast=False):
        return len(self.pred_angles[fast])


    def transform_train(self, img):
        deg = _np.random.uniform(self.pred_angles[False][0], self.pred_angles[False][-1])
        return _spint.rotate(img, deg,
            reshape=False, mode='nearest',
            order=self.order, prefilter=self.prefilter)


    def transform_pred(self, img, i, fast=False):
        return _spint.rotate(img, self.pred_angles[fast][i],
            reshape=False, mode='nearest',
            order=self.order, prefilter=self.prefilter)
