import ml_metrics as metrics


def MAPatK(x,y):
    """
    metrics.mapk.__doc__:

        Computes the mean average precision at k.

        This function computes the mean average prescision at k between two lists
        of lists of items.

        Parameters
        ----------
        x : list
            A list of lists of elements that are to be predicted 
            (order doesn't matter in the lists)
        y : list
            A list of lists of predicted elements
            (order matters in the lists)

        Returns
        -------
        score : double
                The mean average precision at k over the input lists
    """
    return metrics.mapk(x,y)
