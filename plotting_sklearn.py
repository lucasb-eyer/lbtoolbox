#!/usr/bin/env python3

import matplotlib.pyplot as _plt
import sklearn.preprocessing as _skpp
import sklearn.metrics as _skm


def confuse(conf=None, y_pred=None, y_true=None, labels=None, label_order=None, many=None, figsize=5, showpct=None, hidebelow=1e-5, cm=_plt.cm.Spectral_r):
    """
    Specify either `conf` as a matrix of counts,
    or `y_pred` and `y_true` as arrays of labels.

    The optional parameter `labels` can either be a collection, an instance
    of LabelEncoder, or None. If `label_order` is a collection, it defines the
    order in which labels should appear in the matrix.

    If `many` is set to `True`, the confusion matrix will be optimized for
    displaying a large number of classes, i.e. it will contain less details.
    If it is not set, an automatic decision will be attempted.

    If `showpct` is set to `True`, the accuracy of every single cell will be
    shown inside the cell. It defaults to `not many`.

    A different colormap (e.g. for those who like jet =)) can be used via `cm`.
    """

    assert conf is not None or (y_pred is not None and y_true is not None), "Need either `conf` or `y_pred` and `y_true`!"
    assert labels is not None or (y_pred is not None and y_true is not None), "Need either `labels` or `y_pred` and `y_true`!"

    # Turn whatever we have into a LabelEncoder instance.
    LabelEncoder = _skpp.LabelEncoder
    if labels is None and (y_pred is not None and y_true is not None):
        labels = LabelEncoder().fit(list(set(y_pred) | set(y_true)))
    elif not isinstance(labels, LabelEncoder):
        labels = LabelEncoder().fit(labels)

    # Turn whatever we have into a matrix of counts.
    if conf is None:
        conf = _skm.confusion_matrix(y_true, y_pred, labels=labels.transform(labels.classes_))
    #print(conf)

    classes = labels.classes_
    if label_order is not None:
        classes = label_order
        new_order = labels.transform(label_order)
        conf = conf[:,new_order][new_order]

    N = len(labels.classes_)

    # Potentially "auto-determine" whether we have `many` classes or not.
    if many is None:
        many = N > 4*figsize  # This is based on the below N//(4*figsize) stepsize
    if showpct is None:
        showpct = not many

    # Compute relative values of confusion matrix (such that each row sums to 1)
    rconf = _skpp.normalize(conf.astype(float), norm='l1')

    # Draw the confusion matrix itself.
    fig, ax = _plt.subplots(figsize=(figsize, figsize))
    ax.set_aspect(1)
    im = ax.imshow(rconf, cmap=cm, interpolation='nearest')
    # http://stackoverflow.com/a/26720502/2366315
    fig.colorbar(im, fraction=0.046, pad=0.035)

    # Use string labels on the axes.
    step = 1 if not many else N//(4*figsize)
    ax.set_xticks(range(0, N, step))
    ax.set_yticks(range(0, N, step))
    ax.set_xticklabels(classes[::step], rotation='horizontal' if not many else 'vertical')
    ax.set_yticklabels(classes[::step])

    # Fill the boxes with percentage values
    if showpct:
        for x in range(N):
            for y in range(N):
                if rconf[x,y] > hidebelow:
                    ax.annotate("{:.1%}".format(rconf[x,y]), xy=(y,x), ha='center', va='center')

    # Finally, give the average accuracy and the class-mean in the title.
    ax.set_title("Avg. acc: {:.2%}, class mean: {:.2%}".format(conf.trace()/conf.sum(), rconf.trace()/N))

    return fig, ax
