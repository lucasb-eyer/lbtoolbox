#!/usr/bin/env python3

import theano as th
import theano.tensor as T


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


def count_params(model):
    return sum(np.prod(p.get_value().shape) for p in model.params)
