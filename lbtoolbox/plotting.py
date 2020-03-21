#!/usr/bin/env python3
"""My plotting setup

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# Font which got unicode math stuff.
import matplotlib as mpl
mpl.rcParams['font.family'] = 'DejaVu Sans'

# Much more readable plots
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Make NaNs black in my fav. colormaps:
mpl.cm.Spectral.set_bad(color=(0,0,0), alpha=1)
mpl.cm.Spectral_r.set_bad(color=(0,0,0), alpha=1)

# Much better than plt.subplots()
from mpl_toolkits.axes_grid1 import ImageGrid
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from os.path import join as pjoin
from copy import deepcopy
from datetime import datetime
from itertools import chain, repeat, cycle
import numbers
import warnings

from .util import tuplize, flipany
from .evaluation import jaccard_iou, accuracy, mean_precision


try:
    from IPython.display import display, clear_output

    def liveplot(plotfn, *a, savedir=None, savename=None, **kw):
        # See https://github.com/ipython/ipython/issues/7270#issuecomment-355276432
        # TL;DR: Call matplotlib.interactive(False) beforehand.
        fig = plotfn(*a, **kw)
        if fig is None:
            return
        if savedir is not None:
            savename = savename or datetime.now().isoformat()
            assert not isinstance(fig, (tuple, list)), "Not implemented yet!"
            savefig(fig, pjoin(savedir, savename), pdf=False)
        clear_output(True)
        if isinstance(fig, (tuple, list)):
            for f in fig:
                display(f)
        else:
            display(fig)
        plt.close()

except ImportError:
    pass


class LatexLookAlike:
    """Context-manager for making plots inside it look like LaTeX."""
    def __init__(self, fontsize=18, labelpad=4.0):
        self._fontsize = fontsize
        self._labelpad = labelpad

    def __enter__(self):
        self.old_k = deepcopy(mpl.colors.colorConverter.colors['k'])
        mpl.colors.colorConverter.colors['k'] = (0, 0, 0)
        mpl.colors.colorConverter.cache['k'] = (0, 0, 0)

        # with plt.rc_context({...})
        self.old = deepcopy(mpl.rcParams)
        mpl.rcParams.update({
            'font.size': self._fontsize,
            'font.family': 'STIXGeneral',
            #'mathtext.fontset': 'stix',
            'mathtext.fontset': 'cm',
            'text.color': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black',
            'text.color': 'black',
            'axes.labelcolor': 'black',
            'axes.labelpad': self._labelpad,
        })

    def __exit__(self, *a, **kw):
        mpl.colors.colorConverter.colors['k'] = self.old_k
        mpl.colors.colorConverter.cache['k'] = self.old_k
        mpl.rcParams.update(self.old)


def imshow_raw(im, ax=None, shape=None, bgr=False, colordim=2, *args, **kwargs):
    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('cmap', mpl.cm.Spectral_r)

    if shape is not None:
        im = im.reshape(shape)

    if bgr:
        im = flipany(im, colordim)

    if colordim != 2:
        im = np.rollaxis(im, colordim, 3)

    if ax is not None:
        ax.grid(False)
        return ax.imshow(im, *args, **kwargs)
    else:
        figsize = kwargs.pop('figsize', None)
        plt.figure(figsize=figsize)
        plt.grid(False)
        return plt.imshow(im, *args, **kwargs)


# I'm tired of 'fixing' imshow every time.
def imshow(im, ax=None, shape=None, bgr=False, normalize=None, colordim=2, *args, **kwargs):
    kwargs.setdefault('interpolation', 'nearest')

    if shape is not None:
        im = im.reshape(shape)

    if bgr:
        im = flipany(im, colordim)

    if colordim != 2:
        im = np.rollaxis(im, colordim, 3)

    kwargs.setdefault('cmap', mpl.cm.Spectral_r)

    # In the case of a "heatmap," we make it symmetric around 0, unless `normalize` is explicitly passed.
    if im.ndim == 2:
        if normalize is not None and len(normalize) == 2:
            lo, hi = normalize
        else:
            extr = np.nanmax(np.abs(im))
            lo, hi = -extr, extr
        kwargs.setdefault('vmin', lo)
        kwargs.setdefault('vmax', hi)
    # But in the case of RGB images, we need to get them into "regular" range [0-255].
    elif im.ndim == 3 and im.shape[2] == 3 and im.dtype != np.uint8:
        if 0 <= np.nanmin(im) and 10 < np.nanmax(im):
            warnings.warn("Image looks like [0-255] RGB but dtype isn't `uint8`. Renormalizing.")
        if normalize is not None and len(normalize) == 2:
            lo, hi = normalize
        else:
            lo, hi = np.nanmin(im), np.nanmax(im)
        im = np.clip(255*(im.astype(np.float) - lo)/(hi - lo), 0, 255).astype(np.uint8)

    if ax is not None:
        ax.grid(False)
        return ax.imshow(im, *args, **kwargs)
    else:
        figsize = kwargs.pop('figsize', None)
        if figsize is not None:
            plt.figure(figsize=figsize)

        ret = plt.imshow(im, *args, **kwargs)
        if len(im.shape) == 2:
            plt.colorbar()
        plt.grid(False)
        return ret


def lblcmap(n, base_cmap='Accent'):
    """
    Create a colormap of `n` discrete colors (e.g. for semantic segmentation)
    based on the continuous `base_cmap`.
    Also adds a mapping from non-finite inputs to black.
    """
    cmap = plt.cm.get_cmap(base_cmap, n)
    cmap.set_bad(color=(0,0,0), alpha=1)
    cmap.set_bad(color=(0,0,0), alpha=1)
    return cmap


def lblshow(lbl, ax, cmap, **kwargs):
    """
    Shows a labeled image, i.e. an image whose pixels are label-values.
    The colors of the labels is defined by `cmap` (see `lblcmap`) but `-1` is mapped to `nan` first.
    Care is taken so that a colorbar can be shown easily using `lblbar`.
    """
    # Make minus one become NaN, as it usually means "no label here"
    lbl = np.array(lbl, dtype=float)
    lbl[lbl == -1] = np.nan

    return imshow_raw(lbl, ax=ax, cmap=cmap, vmin=-0.5, vmax=cmap.N-0.5)


def lblbar(colorbar, names):
    """
    Adjusts a `colorbar` to show discrete class-`names` as ticks.

    Hint: create the colorbar using either of:
        - `fig.colorbar(im, cax=grid.cbar_axes[0])`
        - `fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.05)`
    """
    cb.set_ticks(np.arange(len(names)))
    cb.set_ticklabels(tuple(names))
    return cb


# I'm also tired of manually making the line in the legend twice as fat as regular, and not-transparent.
def fatlegend(ax=None, *args, **kwargs):
    if ax is not None:
        leg = ax.legend(*args, **kwargs)
    else:
        leg = plt.legend(*args, **kwargs)

    for l in leg.legendHandles:
        l.set_linewidth(l.get_linewidth()*2.0)
        l.set_alpha(1)
    return leg


# TODO: unsure about l, maybe there's a better one!
cm_lab_l = mpl.cm.gray
cm_lab_a = mpl.colors.LinearSegmentedColormap.from_list('CIE a*', [(0, 1, 0, 1), (1, 0, 1, 1)])
cm_lab_b = mpl.colors.LinearSegmentedColormap.from_list('CIE b*', [(0, 0, 1, 1), (1, 1, 0, 1)])

# take any colormap and append a reversed copy of it to itself.
# This way horizontal angles (of 180 and 0) have the same color.
# Don't forget to plot using vmin=0, vmax=2*np.pi when using this!
# See http://matplotlib.org/examples/color/colormaps_reference.html
def cm_angle_from(base):
    return mpl.colors.LinearSegmentedColormap.from_list('angular_' + base.name,
        [base(i) for i in chain(range(base.N), reversed(range(base.N)))])

cm_angle_summer = cm_angle_from(mpl.cm.summer)


def zoomtext(fig_or_axes, factor=2):
    texts = []

    try:
        # It's a figure!
        axes = fig_or_axes.axes
        texts += fig_or_axes.texts
    except AttributeError:
        axes = fig_or_axes

    for ax in tuplize(axes):
        texts += [ax.title, ax.xaxis.label, ax.yaxis.label]
        texts += ax.get_xticklabels()
        texts += ax.get_yticklabels()

    for t in texts:
        t.set_fontsize(round(t.get_fontsize() * factor))


def savefig(fig, name, ticksize=None, pdf=True, **kwargs):
    #kwargs.setdefault('transparent', True)
    kwargs.setdefault('bbox_inches', 'tight')
    #kwargs.setdefault('pad_inches', 0)

    if ticksize is not None:
        if isinstance(ticksize, numbers.Integral):
            majsz = minsz = ticksize
        else:
            majsz, minsz = ticksize
        for ax in fig.axes:
            ax.tick_params(axis='both', which='major', labelsize=majsz)
            ax.tick_params(axis='both', which='major', labelsize=majsz)

    fig.savefig(name + '.png', **kwargs)
    if pdf:
        fig.savefig(name + '.pdf', **kwargs)


# Makes a more-or-less square grid of subplots holding
# len(`what`) axes. kwargs are passed along to mpl's `subplots`.
def imagegrid_for(what, axis=True, ncol=None, figsize=None, **igkw):
    # That's kinda ugly, but I'm not sure of a better and more practical way yet!
    if isinstance(what, tuple) and len(what) == 2 and all(isinstance(w, numbers.Integral) for w in what):
        rows, cols = what
        n = rows*cols
    else:
        try:
            n = len(what)
        except TypeError:
            n = int(what)

        ncol = int(np.ceil(np.sqrt(n))) if ncol is None else ncol
        nrow = int(np.ceil(float(n)/ncol))

    fig = plt.figure(figsize=figsize)
    ig = ImageGrid(fig, 111, (nrow, ncol), **igkw)

    # Remove redundant ones:
    for i in range(1, nrow*ncol - n + 1):
        fig.delaxes(ig.axes_all[-i])

    if not axis:
        for ax in ig.axes_all:
            ax.axis('off')

    return fig, ig


# Makes a more-or-less square grid of subplots holding
# len(`what`) axes. kwargs are passed along to mpl's `subplots`.
def subplotgrid_for(what, axis=True, **kwargs):
    # That's kinda ugly, but I'm not sure of a better and more practical way yet!
    if isinstance(what, tuple) and len(what) == 2 and all(isinstance(w, numbers.Integral) for w in what):
        rows, cols = what
        n = rows*cols
    else:
        try:
            n = len(what)
        except TypeError:
            n = int(what)

        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(float(n)/cols))

    kwargs.setdefault('sharex', True)
    kwargs.setdefault('sharey', True)
    fig, axes = plt.subplots(rows, cols, **kwargs)

    # Remove redundant ones:
    for i in range(1, rows*cols - n + 1):
        fig.delaxes(axes.flat[-i])

    # Necessary to avoid white border when using share[xy]:
    # https://github.com/matplotlib/matplotlib/issues/1789/
    for ax in (axes.flat if n > 1 else [axes]):
        ax.set_adjustable('box-forced')
        if not axis:
            ax.axis('off')

    return fig, axes


def annotline(ax, line, where, fmt=None, xytext=(5,5), halign='left', valign='bottom', markx=False, xhalign='right', xvalign='bottom', linekw={}, textkw={}):
    # Not sure about that orig=True, for now it never made a difference but it sounds better.
    x, y = line.get_data(orig=True)

    # `where` may be a function...
    if callable(where):
        where = where(y)

    # `where` may now be either just a `y` value, or a pair `x, y`
    if isinstance(where, numbers.Number):
        wherey = where
        wherexs = [x[idx] for idx in np.where(y==wherey)[0]]
    else:
        wherey = where[1]
        wherexs = tuplize(where[0])

    # Get the same color as original line if not set expliclitly:
    if 'color' not in linekw:
        linekw = linekw.copy()
        linekw['color'] = line.get_color()
        linekw.setdefault('ls', ':')

    ax.axhline(wherey, **linekw)

    # Get same color as axis tick labels if no color set explicitly:
    if 'color' not in textkw:
        textkw = textkw.copy()
        textkw['color'] = ax.yaxis.get_ticklabels()[0].get_color()

    # Get the formatter from the yaxis major ticks if not set explicilty:
    if fmt is None:
        # For some reason, this doesn't work as well as I'd like yet.
        #txt = ax.yaxis.get_major_formatter().format_data(wherey)
        txt = ax.yaxis.get_major_formatter().format_data_short(wherey)
    else:
        txt = fmt.format(wherey)

    ax.annotate(txt,
        xy=(np.min(x), wherey), xycoords='data',
        xytext=xytext, textcoords='offset points',  # Potential alternative: 'axes fraction'
        ha=halign, va=valign,
    )

    # Potentially add vertical linemarks to mark the x-locations
    if markx:
        for x in wherexs:
            ax.axvline(x, **linekw)
            ax.annotate(ax.yaxis.get_major_formatter().format_data_short(x),
                xy=(x, ax.get_ylim()[0]), xycoords='data',
                xytext=xytext, textcoords='offset points',  # Potential alternative: 'axes fraction'
                ha=xhalign, va=xvalign,
            )


def plot_training(train_errs=None, valid_errs=None, mistakes=None,
                  train_epochs=None, valid_epochs=None, test_epochs=None,
                  ntr=1, nva=1, nte=1, axis=None,
                  besttr=False, bestva=False, bestte=False):
    if axis is None:
        ret = fig, ax = plt.subplots()
    else:
        ret = ax = axis

    trline = None
    if train_errs is not None and len(train_errs) > 1:
        if train_epochs is None:
            train_epochs = np.arange(len(train_errs))
        trline, = ax.plot(train_epochs, np.array(train_errs)/ntr, color='#fb8072', label='Training error')

    valine = None
    if valid_epochs is not None and len(valid_epochs) > 1:
        if valid_epochs is None:
            valid_epochs = np.arange(len(valid_errs))
        valine, = ax.plot(valid_epochs, np.array(valid_errs)/nva, color='#8dd3c7', label='Validation error')

    ax.grid(True)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Errors')
    if ntr > 1 or nva > 1 or nte > 1:
        ax.set_ylabel(ax.get_ylabel() + ' [%]')
        ax.axes.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x*100)))

    teline = None
    if isinstance(mistakes, numbers.Number):
        ax.axhline(mistakes/nte, color='#3b8cc2', linewidth=1, linestyle='--', label='Testing error')
    elif mistakes is not None and len(mistakes) > 1:
        if test_epochs is None and mistakes is not None:
            test_epochs = np.arange(len(mistakes))
        teline, = ax.plot(test_epochs, np.array(mistakes)/nte, color='#3b8cc2', linewidth=1, linestyle='--', label='Testing error')

    # Potentially add lines pinpointing the best points.
    if besttr and trline is not None:
        annotline(ax, trline, np.min, fmt='{:.2%}', markx=True)
    if bestva and valine is not None:
        annotline(ax, valine, np.min, fmt='{:.2%}', markx=True)
    if bestte and teline is not None:
        annotline(ax, teline, np.min, fmt='{:.2%}', markx=True)

    if None not in (trline, valine, teline):
        fatlegend(ax)

    return ret


def plot_cost(train_costs=None, valid_costs=None, cost=None,
              train_epochs=None, valid_epochs=None, test_epochs=None,
              besttr=False, bestva=False, bestte=False,
              axis=None, name='Cost'):
    if axis is None:
        ret = fig, ax = plt.subplots()
    else:
        ret = ax = axis

    trline = None
    if train_costs is not None and len(train_costs) > 1:
        if train_epochs is None:
            train_epochs = np.arange(len(train_costs))
        trline, = ax.plot(train_epochs, np.array(train_costs), color='#fb8072', label='Training cost')

    valine = None
    if valid_costs is not None and len(valid_costs) > 1:
        if valid_epochs is None:
            valid_epochs = np.arange(len(valid_costs))
        valine, = ax.plot(valid_epochs, np.array(valid_costs), color='#8dd3c7', label='Validation cost')

    ax.grid(True)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(name)

    teline = None
    if isinstance(cost, numbers.Number):
        ax.axhline(cost, color='#3b8cc2', linewidth=1, linestyle='--', label='Actual Cost')
    elif cost is not None and len(cost) > 1:
        if test_epochs is None and mistakes is not None:
            test_epochs = np.arange(len(cost))
        teline, = ax.plot(test_epochs, np.array(cost), color='#3b8cc2', linewidth=1, linestyle='--', label='Actual Cost')

    # Potentially add lines pinpointing the best points.
    if besttr and trline is not None:
        annotline(ax, trline, np.min, markx=True)
    if bestva and valine is not None:
        annotline(ax, valine, np.min, markx=True)
    if bestte and teline is not None:
        annotline(ax, teline, np.min, markx=True)

    if None not in (trline, valine, teline):
        fatlegend(ax)

    return ret


def showcounts(*counters, axis=None, sort=None, tickrot='horizontal', percent=True, labels=None, names=None, props=mpl.rcParams['axes.prop_cycle'], legendkw={}):
    """
    - `labels`: A list of names, one for each of the `counters`.
    - `names`: A dict mapping keys from the `counters` to strings to be used as names of those keys.
               If `None`, the keys will be used as names.
    - `percent` can be `False`, `True`, or a number specifying the total by which to divide.
    - `sort` can be:
        - `None` to keep the same order as in the first counter, remaining entries being unsorted.
        - 'name' for alphabetical sorting by key,
        - 'freq' for sorting by frequency in the first counter.
    """
    # Need to make the union of all keys in case some counters don't have some key.
    keys = list(set(chain(*counters)))

    # Either take the keys as names as-is, or map them using passed dict.
    names = keys if names is None else [names[k] for k in keys]

    # Sort alphabetically for comparability if necessary.
    if sort == 'name':
        sortidx = np.argsort(names)
    elif sort == 'freq':
        # Whew lad, probably using the wrong data-structures for this use-case!
        sortidx = [names.index(list(counters[0].keys())[idx]) for idx in np.argsort(list(counters[0].values()))[::-1]]
    else:
        sortidx = [names.index(name) for name in counters[0]]
    names = [names[i] for i in sortidx]
    keys = [keys[i] for i in sortidx]

    # Make all counts equal, inserting 0 for missing keys.
    counts = [np.array([c[k] for k in keys]) for c in counters]

    Nbars = len(names)      # Number of bars per collection
    Ncolls = len(counters)  # Number of collections
    W = 0.8
    margin = 1-W

    if axis is None:
        ret = fig, ax = plt.subplots()
    else:
        ret = ax = axis

    if percent is not False:
        counts = [cnts.astype(float) / (np.sum(cnts) if percent is True else percent) for cnts in counts]
        ax.axes.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x*100)))

    # Plot all the bars, but collect the return values for later legend.
    rects = [
        ax.bar(np.arange(Nbars) + (i+0.5)*W/Ncolls, cnts, W/Ncolls, **prop)
        for i, (cnts, prop) in enumerate(zip(counts, cycle(props)))
    ]

    ax.set_xticks(np.arange(Nbars)+W/2)
    ax.set_xticklabels(names, rotation=tickrot)
    ax.set_xlim(-margin, Nbars-1+W+margin)
    ax.set_ylabel("Occurences" if percent is False else "Frequency [%]")

    if labels is not None:
        assert len(labels) == len(rects), "Number of labels ({}) needs to equal number of collections ({})!".format(len(labels), len(rects))
        ax.legend((r[0] for r in rects), labels, **legendkw)

    return ret


# http://stackoverflow.com/a/16836182
def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mpl.colors.LinearSegmentedColormap('CustomMap', cdict)


# http://stackoverflow.com/a/30689259
def diverge_map(low=(0.094, 0.310, 0.635), high=(0.565, 0.392, 0.173)):
    '''
    low and high are colors that will be used for the two
    ends of the spectrum. they can be either color strings
    or rgb color tuples
    '''
    c = mpl.colors.ColorConverter().to_rgb
    if isinstance(low, basestring): low = c(low)
    if isinstance(high, basestring): high = c(high)
    return make_colormap([low, c('white'), 0.5, c('white'), high])


def linear_map(low=(0.094, 0.310, 0.635), high=(0.565, 0.392, 0.173)):
    '''
    low and high are colors that will be used for the two
    ends of the spectrum. they can be either color strings
    or rgb color tuples
    '''
    c = mpl.colors.ColorConverter().to_rgb
    if isinstance(low, (str, bytes)): low = c(low)
    if isinstance(high, (str, bytes)): high = c(high)
    return make_colormap([low, high])


def confuse(conf, labels, display_order=None, figsize=5, showpct=0.05, rotticks=True, skipticks=None, topticks=True, jacc='l', cm=plt.cm.Spectral_r):
    """
    Renders the confusion matrix as given in absolute counts by `conf`.

    The names of the individual rows/columns of the confusion matrix `conf` are
    given in `labels` and can be filtered and/or reordered by specifying them as
    they should be displayed in `display_order`.

    If `showpct` is set to `True`/`False`, always/never show percentages inside cells.
    Alternativelty, it can be set to a number only showing percentages in those
    cells which are larger-than or equal to that number.

    If `rotticks` is `True` (the default), tick-labels along the X-axis will
    be rotated vertically, which is useful when having many classes.

    When having *very* many classes, it can be useful to show only every
    `skipticks`th tick for legibility, especially if their ordering is fixed.

    `topticks` can be used to switch title and x-ticks to be on bottom and top.

    If `jacc` is either `'l'` or `'r'`, a single column of per-class Jaccard
    intersection over union scores will be shown to the left or right of the
    confusion matrix, respectively. Any other value hides that column.

    A different colormap (e.g. for those who like jet =)) can be used via `cm`.
    """

    # Turn whatever we have into an array of labels
    # if labels is None:
    #     labels = list(set(y_pred) | set(y_true))

    if display_order is not None:
        new_order = [labels.index(l) for l in display_order]
        conf = conf[:,new_order][new_order]
        labels = display_order

    if jacc in ('l', 'r'):
        fig = plt.figure(1, figsize=(figsize, figsize))
        grid = ImageGrid(fig, 111, nrows_ncols=(1,2), axes_pad=0.25, share_all=False, cbar_mode=None)
        ax, axJacc = (grid[0], grid[1]) if jacc == 'r' else (grid[1], grid[0])

        jacc = jaccard_iou(conf)
        axJacc.imshow(jacc[:,None], cmap=cm, interpolation='nearest', vmin=0, vmax=1)
        axJacc.grid(False)
    else:
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        axJacc = None

    # The grid is problematic because it's going through the middle of the bins.
    ax.grid(False)

    # Compute relative values of confusion matrix (such that each row sums to 1)
    rconf = conf / conf.sum(axis=1, keepdims=True)
    im = ax.imshow(rconf, cmap=cm, interpolation='nearest', vmin=0, vmax=1)

    # Use string labels on the axes.
    N = len(labels)
    step = 1+skipticks if skipticks not in (None, False) else 1
    ax.set_xticks(range(0, N, step))
    ax.set_yticks(range(0, N, step))
    ax.set_xticklabels(labels[::step], rotation='vertical' if rotticks else 'horizontal')
    ax.set_yticklabels(labels[::step])
    if topticks:
        ax.xaxis.tick_top()

    if axJacc is not None:
        axJacc.set_xticks([0])
        axJacc.set_xticklabels(['Jaccard IoU'], rotation='vertical' if rotticks else 'horizontal')
        if topticks:
            axJacc.xaxis.tick_top()

    # Fill the boxes with percentage values
    if showpct not in (None, False) and showpct < 1:
        for x in range(N):
            for y in range(N):
                if showpct <= 100*rconf[y,x]:
                    ax.annotate("{:.1%}".format(rconf[y,x]), xy=(x,y), ha='center', va='center')
        if axJacc is not None:
            for y in range(N):
                axJacc.annotate("{:.1%}".format(jacc[y]), xy=(0,y), ha='center', va='center')

    # Finally, give the average accuracy and the class-mean in the title.
    title = "Avg. Jaccard IoU: {:.2%}, Accuracy: {:.2%}, Mean precision: {:.2%}".format(
        np.mean(jaccard_iou(conf)), accuracy(conf), mean_precision(conf)
    )
    if topticks:
        ax.set_title(title, y=-0.15**(figsize**0.25), va='top')  # Magic!!
    else:
        ax.set_title(title, y=1.08)

    return fig, ax
