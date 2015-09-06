#!/usr/bin/env python3

import theano as th
import theano.tensor as T
import numpy as np
import os
from zlib import crc32


def on_gpu(fn):
    if any(x.op.__class__.__name__ in ['Gemv', 'CGemv'] for x in fn.maker.fgraph.toposort()):
        return False
    elif any(x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in fn.maker.fgraph.toposort()):
        return True
    else:
        raise NotImplementedError()


def check_gpu():
    x = T.matrix("x", dtype=th.config.floatX)
    y = T.vector("y", dtype=th.config.floatX)
    z = th.function(inputs=[x, y], outputs=T.dot(x,y))
    return on_gpu(z)


def save_model(model, fname, compress=False, hashmodel=True):
    params, _ = model.parameters()
    kwargs = {"{}-{}".format(i, p.name): p.get_value() for i, p in enumerate(params)}
    if hashmodel:
        fname += '-{}'.format(crc32('\n'.join(sorted(kwargs.keys())).encode('utf-8')))
    if compress:
        np.savez_compressed(fname, **kwargs)
    else:
        np.savez(fname, **kwargs)


def load_model(model, fname, hashmodel=True):
    params, _ = model.parameters()

    if hashmodel:
        pnames = ["{}-{}".format(i, p.name) for i, p in enumerate(params)]
        fname += '-{}'.format(crc32('\n'.join(sorted(pnames)).encode('utf-8')))

    if not fname.endswith(".npz"):
        fname = fname + ".npz"

    with np.load(fname) as ps:
        for i, p in enumerate(params):
            p.set_value(ps["{}-{}".format(i, p.name)])


def count_params(model):
    params, _ = model.parameters()
    return sum(np.prod(p.get_value().shape) for p in params)
