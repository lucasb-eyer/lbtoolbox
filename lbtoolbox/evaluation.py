import numpy as np

def fast_confusion_matrix(y_pred, y_gt, nlabels=None, cm=None):
    """
    NOTE: This can certainly be extended to arbitrary labels without affecting
          speed too much, I just didn't care yet.
    NOTE: This correctly ignores `-1` entries in `y_gt`, but assumes there are
          *only* valid labels in `y_pred`.
    """
    assert cm is not None or nlabels is not None, "Need either pre-allocated `cm` or `nlabels`."

    if cm is None:
        cm = np.zeros((nlabels,nlabels), np.int64)  # Needs to be `int64` not `uint64` for `+=` below to work.
    elif nlabels is None:
        nlabels = cm.shape[0]
    else:
        assert cm.shape == (nlabels, nlabels), "Mismatch between `cm`'s shape and `nlabels`."

    for gt in range(nlabels):
        preds, n = np.unique(y_pred[y_gt == gt], return_counts=True)
        cm[gt, preds] += n
    return cm


def jaccard_iou(cm):
    s = np.sum(cm, axis=0) + np.sum(cm, axis=1) - np.diag(cm)
    return np.diag(cm)/s


def accuracy(cm):
    return cm.trace()/cm.sum()


def mean_precision(cm):
    rcm = cm / np.sum(cm, axis=1, keepdims=True)
    return rcm.trace() / len(cm)
