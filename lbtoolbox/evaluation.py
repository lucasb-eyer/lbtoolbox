def fast_confusion_matrix(y_pred, y_gt, nlabels):
    """
    NOTE: This can certainly be extended to arbitrary labels without affecting
          speed too much, I just didn't care yet.
    NOTE: This correctly ignores `-1` entries in `y_gt`, but assumes there are
          *only* valid labels in `y_pred`.
    """
    cm = np.zeros((nlabels,nlabels), np.uint64)
    for gt in range(nlabels):
        preds, n = np.unique(y_pred[y_gt == gt], return_counts=True)
        cm[gt, preds] = n
    return cm
