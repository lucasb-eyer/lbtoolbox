import os
import json
import numpy as np

from .util import tuplize


def sane_listdir(where, ext='', sortkey=None):
    """
    Intended for internal use.
    Like `os.listdir`, but:
        - Doesn't include hidden files,
        - Always returns results in a sorted order (pass `sortkey=int` for numeric sort),
        - Optionally only return entries whose name ends in `ext`.
    """
    return sorted((i for i in os.listdir(where) if not i.startswith('.') and i.endswith(ext)), key=sortkey)


def create_dat(basename, dtype, shape, fillvalue=None, **meta):
    """ Creates a data file at `basename` and returns a writeable mem-map
        backed numpy array to it.
        Can also be passed any json-serializable keys and values in `meta`.
    """
    Xm = np.memmap(basename, mode='w+', dtype=dtype, shape=shape)
    Xa = np.ndarray.__new__(np.ndarray, dtype=dtype, shape=shape, buffer=Xm)
    #Xa.flush = Xm.flush  # Sadly, we can't just add attributes to a numpy array, need to subclass it.

    if fillvalue is not None:
        Xa.fill(fillvalue)
        #Xa.flush()
        Xm.flush()

    meta.setdefault('dtype', np.dtype(dtype).str)
    meta.setdefault('shape', tuplize(shape))
    json.dump(meta, open(basename + '.json', 'w+'))

    return Xa


def load_dat(basename, mode='r'):
    """ Returns a read-only mem-mapped numpy array to file at `basename`.
    If `mode` is set to `'r+'`, the data can be written, too.
    """
    desc = json.load(open(basename + '.json', 'r'))
    dtype, shape = desc['dtype'], tuplize(desc['shape'])
    Xm = np.memmap(basename, mode=mode, dtype=dtype, shape=shape)
    Xa = np.ndarray.__new__(np.ndarray, dtype=dtype, shape=shape, buffer=Xm)
    #Xa.flush = Xm.flush  # Sadly, we can't just add attributes to a numpy array, need to subclass it.
    return Xa
