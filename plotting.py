#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from itertools import chain


# I'm tired of 'fixing' imshow every time.
def imshow(im, ax=None, cb=True, bgr=False, *args, **kwargs):
    extr = np.max(np.abs(im))
    kwargs.setdefault('vmin', -extr)
    kwargs.setdefault('vmax',  extr)
    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('cmap', mpl.cm.Spectral_r)

    if bgr:
        im = im[:,:,::-1]

    if ax:
        return ax.imshow(im, *args, **kwargs)
    else:
        ret = plt.imshow(im, *args, **kwargs)
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
    for ax in axes.flat:
        ax.set_adjustable('box-forced')
        if not axis:
            ax.axis('off')

    return fig, axes


def show_coefs(coefs, shape=(50,50)):
    # This is hard-coded to look good for the 5-class problem.
    # Would be easy to generalize, but meh.
    K = coefs.shape[0]

    vmax = np.max(np.abs(coefs))
    fig, axes = plt.subplots(nrows=K, ncols=1, figsize=(2, K))
    fig.suptitle('Learned coefficients of the classes', fontsize=16)
    for i, ax in enumerate(axes.flat):
        ax.set_adjustable('box-forced')
        ax.axis('off')
        im = axes[i].imshow(coefs[i,:].reshape(shape), cmap=plt.cm.Spectral, interpolation='nearest', vmin=-vmax, vmax=vmax)
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

