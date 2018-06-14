import tensorflow as tf


def tf_expdec(t, t0, t1, v0, v1):
    """
    Return `v0` until `e` reaches `e0`, then exponentially decay
    to `v1` when `e` reaches `e1` and return `v1` thereafter.
    Copyright (C) 2018 Lucas Beyer - http://lucasb.eyer.be =)
    """
    return tf.train.piecewise_constant(
        t, boundaries=[t0, t1],
        values=[v0, tf.train.exponential_decay(v0, t-t0, t1-t0, v1/v0), v1])
