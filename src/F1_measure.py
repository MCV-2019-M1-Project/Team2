import numpy as np


def F1_measure(gt,res):
    """
    This function computes the F1 measure between gt and res.

    Parameters
    ----------
    gt : Ground truth binary image as numpy array.

    res : Result binary image as numpy array.

    Returns
    -------
    Integer resulting of computing the F1 measure.
    """

    TP = np.sum(np.bitwise_and(gt,res) == 1)
    FN = np.sum(np.bitwise_and(gt,(1-res)) == 1)
    FP = np.sum(np.bitwise_and((1-gt),res) == 1)
    TN = np.sum(np.bitwise_and((1-gt),(1-res)) == 1)

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    return 2*precision*recall/(precision+recall)
