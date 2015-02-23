Lucas' Python Toolbox
=====================

Just a collection of python functions I think I'll need more than once.

The quality varies wildly across functions and I do not intend to make this
usable by anyone besides myself in the foreseeable future. You've been warned.

plotting.py
===========

A bunch of matplotlib plots that I seem to use over and over again.

filterbank.py
=============

Create filterbanks as tuples of `np.array`s. Available filterbanks:

- Difference of oriented Gaussians (DooG)

progress.py
===========

IPython interactive progressbar. TODO: update to new interactive widget.

nnet*
=====

Layers and other stuff for making neural networks with Theano.

This stuff is still highly in flux while I'm figuring out The Right Way to
implement most of this stuff in a as reusable and as orthogonal way as possible.

I will soon add example notebooks and scripts. The following are quick
references to get the big picture.

Training a model
----------------

Create a model class or use one of those in `nnet_models.py`, then:

```
model = MyModel(imshape=(1,28,28), batchsize=1000, npred=10)
model_bgd = lbnn.BatchGradDescent(model)
model_bgd_trainer = lbnn.BGDTrainer(model_bgd)
'{:.3f}M params'.format(lbthutil.count_params(model)/1000000)
model_bgd_trainer.fit(Xtr, ytr, Xva, yva, Xte, yte,
                      lr0=0.01, lr_decrease=2, max_reducs=20, valid_freq=5, test_freq=10,
                      skip_initial=True, skip_final=True)
```

Creating an augmentation pipeline
---------------------------------

Create the pipeline via

```
aug = lbaug.AugmentationPipeline((32,32), Xtr, ytr,
    lbaug.HorizontalFlipper(),
    lbaug.Rotator(0, 180, 5),
    lbaug.Cropper((28,28))
)
```

then pass it to the `fit` function as parameter `aug=aug` and, if it includes a
resizing augmentation, be sure to create the model using `aug.outshape()` as
image-shape, not the original image-shape.

thutil.py
=========

Minor utilities for working with Theano. Currently checks for GPU.
