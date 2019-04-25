#!/usr/bin/env python3

import numpy as _np

import contextlib
import numbers
import signal
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


def batched(batchsize, *arrays, **kw):
    """
    A generator function which goes through all of `arrays` together,
    but in batches of size `batchsize` along the first dimension.

    batched(3, np.arange(10), np.arange(10))

    will yield sub-arrays of the given ones four times, the fourth one only
    containing a single value.

    Valid keyword arguments:

    - `shuf`: Shuffle the whole dataset before going through it.
       Default: `False`.
    - `shuf_batches`: Batch the data and go through the batches in random order.
       This is useful e.g. if the data is sorted by length, to have batches only
       contain data of the same length, but go through lenghts randomly.
       Default: `False`.
    - `droplast`: Do not return the last batch if it's smaller than `batchsize`.
       Default: `False`.
    - `N`: upper bound to the total number of returned entries (not batches).
       Default: `None`, meaning batch through everything.
    - `progress`: A function to call for progress updates, fn(i, N).
    """

    shuf = kw.get('shuf', False)
    shuf_batches = kw.get('shuf_batches', False)
    droplast = kw.get('droplast', False)
    N = kw.get('N', None)
    progress = kw.get('progress', lambda i, N: None)

    assert(len(arrays) > 0)

    # Shorthands
    n = len(arrays[0])
    bs = batchsize

    # Assumption: all arrays have the same 1st dimension as the first one.
    assert(all(len(x) == n if N is None else len(x) >= N for x in arrays))

    if shuf is not False:
        indices = check_random_state(shuf).permutation(n)
    elif shuf_batches is not False:
        indices = _np.arange(n)
        batch_indices, last = _np.array(indices[:n-n%bs:bs]), indices[n-n%bs::bs]
        check_random_state(shuf_batches).shuffle(batch_indices)
        indices = _np.concatenate([indices[i:i+bs] for i in _np.r_[batch_indices, last]])
    else:
        indices = _np.arange(n)

    # Now, cut off at `N`, if specified.
    if N is not None:
        n = min(n, N)

    # First, go through all full batches.
    for i in range(n // batchsize):
        yield maybetuple(_fancyidx(x, indices[i*batchsize:(i+1)*batchsize]) for x in arrays)
        progress(min((i+1)*batchsize, len(indices)), len(indices))

    # And now maybe return the last batch.
    rest = n % batchsize
    if rest != 0 and not droplast:
        i = (n//batchsize)*batchsize
        yield maybetuple(_fancyidx(x, indices[i:n]) for x in arrays)
        progress(len(indices), len(indices))


try:
    import h5py as _h5
    _has_h5py = True
except ImportError:
    _has_h5py = False

# A work-around for supporting fancy-indexing for numpy lists/tuples.
def _fancyidx(x, idx):
    if isinstance(x, _np.ndarray):
        return x[idx]
    elif _has_h5py and isinstance(x, _h5.Dataset):
        # Indices for fancy-indexing need to be sorted here!
        sort = _np.argsort(idx)
        return x[idx[sort].tolist()][sort]
    else:
        return [x[i] for i in idx]


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

    - If `seed` is `None` or `True`, return the `RandomState` singleton used by `np.random`.
    - If `seed` is an `int`, return a new `RandomState` instance seeded with `seed`.
    - If `seed` is already a `RandomState` instance, return it.
    - Otherwise raise `ValueError`.
    """
    if seed in (None, True, _np.random):
        return _np.random.mtrand._rand

    if isinstance(seed, (numbers.Integral, _np.integer)):
        return _np.random.RandomState(seed)

    if isinstance(seed, _np.random.RandomState):
        return seed

    raise ValueError('{!r} cannot be used to seed a numpy.random.RandomState instance'.format(seed))


def writenow(f, fmt, *args, **kwargs):
    """
    Formats the string `fmt` with the given `args` and `kwargs`, then writes it
    to `f` and flushes f.

    If `f` is `None`, nothing is done.
    """
    if f is None:
        return

    f.write(fmt.format(*args, **kwargs))
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


# Based on https://gist.github.com/nonZero/2907502
class Uninterrupt(object):
    """
    Use as:

    with Uninterrupt() as u:
        while not u.interrupted:
            # train
    """
    def __init__(self, sigs=[signal.SIGINT], verbose=False):
        self.sigs = sigs
        self.verbose = verbose
        self.interrupted = False
        self.orig_handlers = None

    def __enter__(self):
        if self.orig_handlers is not None:
            raise ValueError("Can only enter `Uninterrupt` once!")

        self.interrupted = False
        self.orig_handlers = [signal.getsignal(sig) for sig in self.sigs]

        def handler(signum, frame):
            self.release()
            self.interrupted = True
            if self.verbose:
                print("Interruption scheduled...", flush=True)

        for sig in self.sigs:
            signal.signal(sig, handler)

        return self

    def __exit__(self, type_, value, tb):
        self.release()

    def release(self):
        if self.orig_handlers is not None:
            for sig, orig in zip(self.sigs, self.orig_handlers):
                signal.signal(sig, orig)
            self.orig_handlers = None


class BackgroundFunction:
    def __init__(self, function, prefetch_count, reseed=reseed, **kwargs):
        """Parallelize a function to prefetch results using mutliple processes.
        Args:
            function: Function to be executed in parallel.
            prefetch_count: Number of samples to prefetch.
            resee: Function to call in each worker at init (usually to re-seed the RNG).
            kwargs: Keyword args passed to the executed function.

        I stole and extended this from:
        https://github.com/Pandoro/tools/blob/master/utils.py
        so we are even now :)
        """
        self.function = function
        self.prefetch_count = prefetch_count
        self.kwargs = kwargs
        self.output_queue = multiprocessing.Queue(maxsize=prefetch_count)
        self.procs = []
        for i in range(self.prefetch_count):
            p = multiprocessing.Process(
                target=BackgroundFunction._compute_next,
                args=(self.function, self.kwargs, self.output_queue, reseed))
            p.daemon = True  # To ensure it is killed if the parent dies.
            p.start()
            self.procs.append(p)

    def fill_status(self, normalize=False):
        """Returns the fill status of the underlying queue.
        Args:
            normalize: If set to True, normalize the fill status by the max
                queue size. Defaults to False.
        Returns:
            The possibly normalized fill status of the underlying queue.
        """
        return self.output_queue.qsize() / (self.prefetch_count if normalize else 1)

    def __call__(self):
        """Obtain one of the prefetched results or wait for one.
        Returns:
            The output of the provided function and the given keyword args.
        """
        output = self.output_queue.get(block=True)
        return output

    def __del__(self):
        """Signal the processes to stop and join them."""
        for p in self.procs:
            p.terminate()
            p.join()

    def _compute_next(function, kwargs, output_queue, reseed):
        """Helper function to do the actual computation in a non_blockig way.
        Since this will always run in a new process, we ignore the interrupt
        signal for the processes. This should be handled by the parent process
        which kills the children when the object is deleted.
        Some more discussion can be found here:
        https://stackoverflow.com/questions/1408356/keyboard-interrupts-with-pythons-multiprocessing-pool
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # By default the random state is copied across processes.
        reseed()
        while True:
            output_queue.put(function(**kwargs))


def smooth(a, w, head=True, tail=True):
    """ Matlab-like smooth

    a: 1D array to be smoothed
    w: smoothing window size, must be odd number.

    Credit goes to https://stackoverflow.com/a/40443565/2366315
    """
    assert w % 2 == 1, 'Need odd window size!'
    if len(a) < w:
        return a  # TODO: Do something else sensible?
    smoothed = _np.convolve(a, _np.ones(w, int), 'valid') / w

    r = _np.arange(1, w-1, 2)
    if head:
        head = _np.cumsum(a[:w-1])[::2]/r
        smoothed = _np.r_[head, smoothed]

    if tail:
        tail = (_np.cumsum(a[:-w:-1])[::2]/r)[::-1]
        smoothed = _np.r_[smoothed, tail]

    return smoothed


def smooth2(y, w, head=False, tail=False):
    x0 = 0 if head else w//2
    x1 = len(y) if tail else len(y) - w//2
    return np.arange(x0, x1), smooth(y, w)[x0:x1]


def truncrandn_approx(lo, hi, *dims):
    """
    Sample values as specified by `dims` from normal distribution truncated to `[lo, hi]`.
    The result is very similar to re-sampling values that fall outside the range,
    but in bounded time, and faster.
    """
    std = (hi-lo)/2.0/3.0  # This std will result in 99.7% of samples
    mean = (lo + hi)/2.0
    r = std*_np.random.randn(*dims)+mean

    if len(dims) == 0:
        if lo <= r <= hi:
            return r
        else:
            # TODO: Might also do some module-stuff to avoid re-sampling!
            return _np.random.uniform(lo, hi)
    else:
        outside = (r < lo) | (hi < r)
        r[outside] = _np.random.uniform(lo, hi, sum(outside))
        return r


def floorlog10(v):
    """Round down to the nearest number in the current power of ten.

    Useful for limits in log-axis plots.

    floorlog10(98.3) -> 90.0
    floorlog10(11.5) -> 10.0
    floorlog10(9.2) -> 9.0
    floorlog10(0.123) -> 0.1
    """
    n = _np.log10(v)
    n = _np.floor(n)
    n = 10**n
    return (v//n)*n


def ceillog10(v):
    """Round up to the nearest number in the current power of ten.

    Useful for limits in log-axis plots.

    floorlog10(98.3) -> 100.0
    floorlog10(11.5) -> 20.0
    floorlog10(9.2) -> 10.0
    floorlog10(0.123) -> 0.2
    """
    n = _np.log10(v)
    n = _np.floor(n)
    n = 10**n
    return (v//n+1)*n


def randints(los, his, shape=1, dtype='l'):
    """
    Like np.random.randint (i.e. [los,his) domain) but a whole array of them.
    Probably not the most efficient implementation, but good enough for now.

    Returns array of shape (len(los), *shape)
    """
    assert len(los) == len(his)

    if not isinstance(shape, tuple):
        shape = (shape,) if shape != 1 else tuple()

    x = _np.empty((len(los),) + shape, dtype)
    for i, (lo, hi) in enumerate(zip(los, his)):
        x[i] = _np.random.randint(lo, hi, shape, dtype)
    return x


def create_dataset_like(g, name, other, **kwupdate):
    kw = {k: kwupdate.get(k, getattr(other, k)) for k in [
        'shape', 'dtype', 'chunks', 'maxshape', 'compression',
        'compression_opts', 'scaleoffset', 'shuffle', 'fletcher32', 'fillvalue']
    }

    # Avoid 
    if kw['maxshape'] == kw['shape']:
        del kw['maxshape']

    return g.create_dataset(name, **kw)


def ramp(e, e0, v0, e1, v1):
    """
    Return `v0` until `e` reaches `e0`, then linearly interpolate
    to `v1` when `e` reaches `e1` and return `v1` thereafter.

    Copyright (C) 2017 Lucas Beyer - http://lucasb.eyer.be =)
    """
    if e < e0:
        return v0
    elif e < e1:
        return v0 + (v1-v0)*(e-e0)/(e1-e0)
    else:
        return v1


def expdec(e, e0, v0, e1, v1, eNone=float('inf')):
    """
    Return `v0` until `e` reaches `e0`, then exponentially decay
    to `v1` when `e` reaches `e1` and return `v1` thereafter, until
    reaching `eNone`, after which it returns `None`.

    Copyright (C) 2017 Lucas Beyer - http://lucasb.eyer.be =)
    """
    if e < e0:
        return v0
    elif e < e1:
        return v0 * (v1/v0)**((e-e0)/(e1-e0))
    elif e < eNone:
        return v1
    else:
        return None


def stairs(e, v, *evs):
    """ Implements a typical "stairs" schedule for learning-rates.
    Best explained by example:

    stairs(e, 0.1, 10, 0.01, 20, 0.001)

    will return 0.1 if e<10, 0.01 if 10<=e<20, and 0.001 if 20<=e
    """
    for e0, v0 in zip(evs[::2], evs[1::2]):
        if e < e0:
            break
        v = v0
    return v
