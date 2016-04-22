#!/usr/bin/env python3

import matplotlib.pyplot as _plt
import sklearn.metrics as _skm


def confuse(conf=None, y_pred=None, y_true=None, labels=None, display_order=None, figsize=5, showpct=0.05, rotticks=True, skipticks=None, topticks=True, cm=_plt.cm.Spectral_r):
    """
    Specify either `conf` as a matrix of counts,
    or `y_pred` and `y_true` as arrays of labels.

    The optional parameter `labels` can either be a collection, an instance
    of LabelEncoder, or None. If `display_order` is a collection, it defines the
    order in which labels should appear in the matrix.

    If `showpct` is set to `True`, always show percentages inside all cells.
    Alternativelty, it can be set to a number only showing percentages in those
    cells which are larger-than or equal to that number.

    If `rotticks` is `True` (the default), tick-labels along the X-axis will
    be rotated vertically, which is useful when having many classes.

    When having *very* many classes, it can be useful to show only every
    `skipticks`th tick for legibility, especially if their ordering is fixed.

    `topticks` can be used to switch title and x-ticks to be on bottom and top.

    A different colormap (e.g. for those who like jet =)) can be used via `cm`.
    """

    assert conf is not None or (y_pred is not None and y_true is not None), "Need either `conf` or `y_pred` and `y_true`!"
    assert labels is not None or (y_pred is not None and y_true is not None), "Need either `labels` or `y_pred` and `y_true`!"

    # Turn whatever we have into an array of labels
    if labels is None and (y_pred is not None and y_true is not None):
        labels = list(set(y_pred) | set(y_true))

    # Turn whatever we have into a matrix of counts.
    if conf is None:
        conf = _skm.confusion_matrix(y_true, y_pred, labels=labels)

    if display_order is not None:
        new_order = [labels.index(l) for l in display_order]
        conf = conf[:,new_order][new_order]
        labels = display_order

    N = len(labels)

    # Compute relative values of confusion matrix (such that each row sums to 1)
    rconf = conf / conf.sum(axis=1, keepdims=True)

    # Draw the confusion matrix itself.
    fig, ax = _plt.subplots(figsize=(figsize, figsize))
    ax.set_aspect(1)
    im = ax.imshow(rconf, cmap=cm, interpolation='nearest', vmin=0, vmax=1)
    # http://stackoverflow.com/a/26720502/2366315
    fig.colorbar(im, fraction=0.046, pad=0.035)

    # Use string labels on the axes.
    step = 1+skipticks if skipticks not in (None, False) else 1
    ax.set_xticks(range(0, N, step))
    ax.set_yticks(range(0, N, step))
    ax.set_xticklabels(labels[::step], rotation='vertical' if rotticks else 'horizontal')
    ax.set_yticklabels(labels[::step])

    if topticks:
        ax.xaxis.tick_top()

    # Fill the boxes with percentage values
    if showpct not in (None, False) and showpct < 1:
        for x in range(N):
            for y in range(N):
                if showpct <= 100*rconf[x,y]:
                    ax.annotate("{:.1%}".format(rconf[x,y]), xy=(y,x), ha='center', va='center')

    # Finally, give the average accuracy and the class-mean in the title.
    avgacc = conf.trace()/conf.sum()
    clmean = rconf.trace()/N
    title = "Avg. acc: {:.2%}, class mean: {:.2%}".format(avgacc, clmean)
    if topticks:
        ax.set_title(title, y=-0.15**(figsize**0.25), va='top')  # Magic!!
    else:
        ax.set_title(title, y=1.08)

    # The grid is problematic because it's going through the middle of the bins.
    ax.grid(False)

    return fig, ax
