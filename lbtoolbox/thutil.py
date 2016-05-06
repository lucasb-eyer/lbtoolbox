#!/usr/bin/env python3
# coding: utf-8

import theano as th
import theano.tensor as T
import numpy as np
import os
from zlib import crc32

from .util import printnow


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


def hasnan(model, params=True, grads=True):
    nans = []
    for p, g in zip(*model.parameters()):
        # TODO: If this ever turns into a bottleneck, use Bottleneck! (py lib)
        if params and np.isnan(np.sum(p.get_value())):
            nans.append(p)
        if grads and np.isnan(np.sum(g.get_value())):
            nans.append(g)
    return nans


def showstats(net, param=True, grad=True):
    params, grads = net.parameters()
    maxlen = max(len(p.name) for p in params)
    if param:
        for p in params:
            v = p.get_value()
            fmt = "{name:{namew}} σ={std:.4f}"
            kw = {'name': p.name + ':', 'namew': maxlen+1, 'std': np.std(v)}
            if p.ndim == 2:
                fmt += ', ρ={specrad:.4f}'
                kw['specrad'] = max(abs(np.linalg.svd(v, compute_uv=False)))
            printnow(fmt, **kw)

    if grad:
        for p, g in zip(params, grads):
            v = g.get_value()
            printnow("{:{namew}} ∈ [{: .5f},{: .5f}]", p.name, np.min(v), np.max(v), namew=maxlen)
