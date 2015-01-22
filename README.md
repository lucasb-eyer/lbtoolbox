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

nnet.py
=======

Layers and other stuff for making neural networks with Theano.
Might overlap with pylearn2 a lot, still need to learn it!

thutil.py
=========

Minor utilities for working with Theano. Currently checks for GPU.
