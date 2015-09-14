#!/usr/bin/env python3

import numpy as _np

import contextlib
import numbers
import sys


def tuplize(what, lists=True, tuplize_none=False):
    """
    If `what` is a tuple, return it as-is, otherwise put it into a tuple.
    If `lists` is true, also consider lists to be tuples (the default).
    If `tuplize_none` is true, a lone `None` results in an empty tuple,
    otherwise it will be returned as `None` (the default).
    """
    if what is None:
        if tuplize_none:
            return tuple()
        else:
            return None

    if isinstance(what, tuple) or (lists and isinstance(what, list)):
        return tuple(what)
    else:
        return (what,)


def maybetuple(what):
    """
    Transforms `what` into a tuple, except if it's of length one, then it's
    returned as-is, or if it's of length zero, then `None` is returned.
    """
    t = tuple(what)
    return t if len(t) > 1 else t[0] if len(t) == 1 else None


def collect(what, drop_nones=True):
    """
    Returns a tuple that is the concatenation of all tuplized things in `what`.
    """
    return sum((tuplize(w, tuplize_none=drop_nones) for w in what), tuple())


# Thanks http://stackoverflow.com/a/2891805/2366315
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    """
    Temporarily set numpy print options.
    Accepts same options as http://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html

    Example:

        with printoptions(precision=3, suppress=True):
            print(array)
    """
    original = _np.get_printoptions()
    _np.set_printoptions(*args, **kwargs)
    yield
    _np.set_printoptions(**original)


@contextlib.contextmanager
def printshort(fmt='{: 0.3f}'):
    """
    Same as `printoptions` with default args for printing in very short
    and fixed float format.
    """
    original = _np.get_printoptions()
    _np.set_printoptions(formatter={'float': fmt.format})
    yield
    _np.set_printoptions(**original)


def batched(batchsize, *arrays, shuf=False, droplast=False):
    """
    A generator function which goes through all of `arrays` together,
    but in batches of size `batchsize` along the first dimension.

    batched_padded(3, np.arange(10), np.arange(10))

    will yield sub-arrays of the given ones four times, the fourth one only
    containing a single value.
    """

    assert(len(arrays) > 0)

    n = arrays[0].shape[0]

    # Assumption: all arrays have the same 1st dimension as the first one.
    assert(all(x.shape[0] == n for x in arrays))

    indices = _np.arange(n)
    if shuf is not False:
        rng = check_random_state(shuf)
        rng.shuffle(indices)

    # First, go through all full batches.
    for i in range(n // batchsize):
        yield maybetuple(x[indices[i*batchsize:(i+1)*batchsize]] for x in arrays)

    # And now maybe return the last batch.
    rest = n % batchsize
    if rest != 0 and not droplast:
        yield maybetuple(x[indices[-rest:]] for x in arrays)


# Pads the first dimension of `x` to length `l`
def _pad_first_dim_to(x, l):
    pads = ((0, l - x.shape[0]),) + tuple((0,0) for _ in range(len(x.shape)-1))
    return _np.pad(x, pads, mode='constant')


def batched_padded(batchsize, *args):
    """
    A generator function which goes through all of `args` together,
    but in batches of size `batchsize` along the first dimension,
    and possibly pads the last batch if necessary.
    Yields the batchsize as first entry of the tuple.

    batched_padded(3, np.arange(10), np.arange(10))

    will yield sub-arrays of the given ones four times.
    """

    assert(len(args) > 0)

    n = args[0].shape[0]

    # Assumption: all args have the same 1st dimension as the first one.
    assert(all(x.shape[0] == n for x in args))

    # First, go through all full batches.
    for i in range(n // batchsize):
        yield (batchsize,) + tuple(x[i*batchsize:(i+1)*batchsize] for x in args)

    # And now maybe return the last, padded batch.
    rest = n % batchsize
    if rest != 0:
        start = (n//batchsize)*batchsize
        yield (rest,) + tuple(_pad_first_dim_to(x[start:], batchsize) for x in args)


def batched_padded_x(batchsize, x, y):
    """
    Like `batched_padded` for `x`, but doesn't pad the last `y`, in effect
    returning a last `y` which might be shorter than `batchsize`.

    Thus, because the `y` carries the batch's size, no size is returned anymore.
    """
    n = x.shape[0]

    assert(y.shape[0] == n)

    # First, go through the full batches.
    for i in range(n // batchsize):
        yield tuple(x[i*batchsize:(i+1)*batchsize] for x in (x,y))

    # And now maybe return the last, x padded and y non-padded, batch.
    if n % batchsize != 0:
        start = (n//batchsize)*batchsize
        yield _pad_first_dim_to(x[start:], batchsize), y[start:]


# Blatantly "inspired" by sklearn, for when that's not available.
def check_random_state(seed):
    """
    Turn `seed` into a `np.random.RandomState` instance.

    - If `seed` is `None`, return the `RandomState` singleton used by `np.random`.
    - If `seed` is an `int`, return a new `RandomState` instance seeded with `seed`.
    - If `seed` is already a `RandomState` instance, return it.
    - Otherwise raise `ValueError`.
    """
    if seed is None or seed is _np.random:
        return _np.random.mtrand._rand

    if isinstance(seed, (numbers.Integral, _np.integer)):
        return _np.random.RandomState(seed)

    if isinstance(seed, _np.random.RandomState):
        return seed

    raise ValueError('{!r} cannot be used to seed a numpy.random.RandomState instance'.format(seed))


def writenow(f, fmt, *args, **kwargs):
    """
    Formats the string `fmt` with the given `args` and `kwargs`, then writes it
    (including a final newline) to `f` and flushes f.

    If `f` is `None`, nothing is done.
    """
    if f is None:
        return

    f.write(fmt.format(*args, **kwargs))
    if not fmt.endswith('\n'):
        f.write('\n')
    f.flush()


def printnow(fmt, *args, **kwargs):
    """
    Formats the string `fmt` with the given `args` and `kwargs`, then prints
    and flushes it to stdout.
    """
    return writenow(sys.stdout, fmt, *args, **kwargs)


def flipany(a, dim):
    """
    `flipany(a, 0)` is equivalent to `flipud(a)`,
    `flipany(a, 1)` is equivalent to `fliplr(a)` and the rest follows naturally.
    """
    # Put the axis in front, flip that axis, then move it back.
    return _np.swapaxes(_np.swapaxes(a, 0, dim)[::-1], 0, dim)
