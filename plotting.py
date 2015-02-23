#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from itertools import chain


# I'm tired of 'fixing' imshow every time.
def imshow(im, ax=None, shape=None, bgr=False, normalize=None, *args, **kwargs):
    kwargs.setdefault('interpolation', 'nearest')

    if shape is not None:
        im = im.reshape(shape)

    if bgr:
        im = im[:,:,::-1]

    if normalize is True:
        im = (255*(im.astype(np.float) - np.min(im))/(np.max(im)-np.min(im))).astype(np.uint8)
    elif normalize is not None and len(normalize) == 2:
        im = (255*(im.astype(np.float) - normalize[0])/(normalize[1]-normalize[0])).astype(np.uint8)

    # Spectral colormap only if it's not a color image
    if len(im.shape) == 2:
        kwargs.setdefault('cmap', mpl.cm.Spectral_r)
        extr = np.max(np.abs(im))
        kwargs.setdefault('vmin', -extr)
        kwargs.setdefault('vmax',  extr)

    if ax is not None:
        return ax.imshow(im, *args, **kwargs)
    else:
        ret = plt.imshow(im, *args, **kwargs)
        if len(im.shape) == 2:
            plt.colorbar()
        return ret

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

    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(float(n)/rows))

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


def show_coefs(coefs, shape, names=None):
    fig, axes = subplotgrid_for(coefs, axis=False)
    fig.suptitle('Learned coefficients of the classes', fontsize=16)

    for coef, ax, name in zip(coefs, axes.flat, names or [None]*len(coefs)):
        if name is not None:
            ax.set_title(name)
        im = imshow(coef, ax, shape=shape)
    fig.colorbar(im, ax=axes.ravel().tolist())

    return fig, axes


def plot_training(train_errs, valid_errs, score=None):
    fig, ax = plt.subplots()

    ax.plot(train_errs, color='#fb8072', label='Training error')
    ax.plot(valid_errs, color='#8dd3c7', label='Validation error')

    ax.grid(True)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error [%]')
    ax.axes.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x*100)))

    if score:
        ax.axhline(1-score, color='#3b8cc2', linewidth=1, linestyle='--', label='Testing error')

    ax.legend()

    return fig, ax

def plot_cost(train_costs, valid_costs, cost=None):
    fig, ax = plt.subplots()

    ax.plot(train_costs, color='#fb8072', label='Training cost')
    ax.plot(valid_costs, color='#8dd3c7', label='Validation cost')

    ax.grid(True)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cost')

    if cost:
        ax.axhline(cost, color='#3b8cc2', linewidth=1, linestyle='--', label='Solution cost')

    ax.legend()

    return fig, ax

