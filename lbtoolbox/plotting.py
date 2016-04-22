#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from itertools import chain, repeat, cycle
import numbers

from .util import tuplize, flipany


try:
    from IPython.display import display, clear_output

    def liveplot(plotfn, *a, **kw):
        clear_output(True)
        display(plotfn(*a, **kw))
        plt.close()

except ImportError:
    pass


# I'm tired of 'fixing' imshow every time.
def imshow(im, ax=None, shape=None, bgr=False, normalize=None, colordim=2, *args, **kwargs):
    kwargs.setdefault('interpolation', 'nearest')

    if shape is not None:
        im = im.reshape(shape)

    if bgr:
        im = flipany(im, colordim)

    if colordim != 2:
        im = np.rollaxis(im, colordim, 3)

    if normalize is True:
        im = (255*(im.astype(np.float) - np.min(im))/(np.max(im)-np.min(im))).astype(np.uint8)
    elif normalize is not None and len(normalize) == 2:
        im = (255*(im.astype(np.float) - normalize[0])/(normalize[1]-normalize[0])).astype(np.uint8)

    # Spectral colormap only if it's not a color image and no map is given.
    if len(im.shape) == 2:
        kwargs.setdefault('cmap', mpl.cm.Spectral_r)
        extr = np.max(np.abs(im))
        kwargs.setdefault('vmin', -extr)
        kwargs.setdefault('vmax',  extr)

    if ax is not None:
        ax.grid(False)
        return ax.imshow(im, *args, **kwargs)
    else:
        ret = plt.imshow(im, *args, **kwargs)
        if len(im.shape) == 2:
            plt.colorbar()
        plt.grid(False)
        return ret


# I'm also tired of manually making the line in the legend twice as fat as regular.
def fatlegend(ax, *args, **kwargs):
    leg = ax.legend(*args, **kwargs)
    for l in leg.legendHandles:
        l.set_linewidth(l.get_linewidth()*2.0)
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


def savefig(fig, name, **kwargs):
    kwargs.setdefault('transparent', True)
    kwargs.setdefault('bbox_inches', 'tight')

    fig.savefig(name + '.png', **kwargs)
    fig.savefig(name + '.pdf', **kwargs)


# Makes a more-or-less square grid of subplots holding
# len(`what`) axes. kwargs are passed along to mpl's `subplots`.
def subplotgrid_for(what, axis=True, **kwargs):
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


def show_coefs(coefs, shape, names=repeat(None)):
    fig, axes = subplotgrid_for(coefs, axis=False)
    fig.suptitle('Learned coefficients of the classes', fontsize=16)

    for coef, ax, name in zip(coefs, axes.flat, names):
        if name is not None:
            ax.set_title(name)
        im = imshow(coef, ax, shape=shape)
    fig.colorbar(im, ax=axes.ravel().tolist())

    return fig, axes


def annotline(ax, line, where, fmt=None, xytext=(5,5), halign='left', valign='bottom', markx=False, linekw={}, textkw={}):
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
        xy=(x[0], wherey), xycoords='data',
        xytext=xytext, textcoords='offset points',  # Potential alternative: 'axes fraction'
        ha=halign, va=valign,
    )

    # Potentially add vertical linemarks to mark the x-locations
    if markx:
        for x in wherexs:
            ax.axvline(x, **linekw)


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


def showcounts(*counters, axis=None, asort=True, tickrot='horizontal', percent=True, labels=None, props=mpl.rcParams['axes.prop_cycle'], legendkw={}):
    # Need to make the union of all keys in case some counters don't have some key.
    names = np.array(list(set(chain(*counters))))

    # Sort alphabetically for comparability if necessary.
    if asort:
        names = np.sort(names)

    # Make all counts equal, inserting 0 for missing keys.
    counts = [np.array([c[k] for k in names]) for c in counters]

    Nbars = len(names)      # Number of bars per collection
    Ncolls = len(counters)  # Number of collections
    W = 0.8
    margin = 1-W

    if axis is None:
        ret = fig, ax = plt.subplots()
    else:
        ret = ax = axis

    if percent:
        counts = [cnts.astype(float) / np.sum(cnts) for cnts in counts]
        ax.axes.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x*100)))

    # Plot all the bars, but collect the return values for later legend.
    rects = [
        ax.bar(np.arange(Nbars) + i*W/Ncolls, cnts, W/Ncolls, **prop)
        for i, (cnts, prop) in enumerate(zip(counts, cycle(props)))
    ]

    ax.set_xticks(np.arange(Nbars)+W/2)
    ax.set_xticklabels(names, rotation=tickrot)
    ax.set_xlim(-margin, Nbars-1+W+margin)
    ax.set_ylabel("Frequency [%]" if percent else "Occurences")

    if labels is not None:
        assert len(labels) == len(rects), "Number of labels needs to equal number of collections!"
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
