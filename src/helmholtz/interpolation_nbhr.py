"""Fits interpolation to 1D Helmholtz test functions in particular (with specific coarse neighborhoods)."""
import numpy as np
import scipy.linalg
import sklearn.metrics.pairwise
from numpy.linalg import norm

_SMALL_NUMBER = 1e-15


def geometric_neighbors(n: int, w: int, nc: int):
    # Here there are aggregate_shape points per aggregate and the same number of coarse vars per aggregate, but in
    # general aggregate sizes may vary.
    fine_var = np.arange(0, n, dtype=int)
    # Index of aggregate that i belongs to, for each i in fine_var.
    aggregate_of_fine_var = fine_var // w
    num_aggregates = n // w
    num_coarse_vars = nc * num_aggregates
    # Global index of coarse variables with each aggregate.
    aggregate_coarse_vars = np.array([np.arange(ic, ic + nc, dtype=int)
                                      for ic in range(0, num_coarse_vars, nc)])
    nbhr = [None] * w
    for fine_var in range(w):
        center_aggregate = fine_var // w
        if fine_var < w // 2:
            # Use center aggregate and left neighboring aggregate.
            nbhr_aggregate = (center_aggregate - 1) % num_aggregates
        else:
            # Use center aggregate and left neighboring aggregate.
            nbhr_aggregate = (center_aggregate + 1) % num_aggregates
#        coarse_nbhrs = np.union1d(aggregate_coarse_vars[center_aggregate], aggregate_coarse_vars[nbhr_aggregate])
        coarse_nbhrs = aggregate_coarse_vars[center_aggregate]
        print(fine_var, center_aggregate, nbhr_aggregate, coarse_nbhrs)
        nbhr[fine_var] = coarse_nbhrs
    return nbhr


def sort_neighbors_by_similarity(x_aggregate: np.array, xc: np.array, nbhr: np.array):
    return np.array([nbhr_of_i[np.argsort(-similarity(x_aggregate[:, i][:, None], xc[:, nbhr_of_i]))][0]
                     for i, nbhr_of_i in enumerate(nbhr)])


def similarity(x, xc):
    """Returns all pairwise correlation similarities between x and xc.

    ***DOES NOT ZERO OUT MEAN, which assumes intercept = False in fitting interpolation.***
    """
    # Calculate all pairwise correlation similarities between x and xc. Zero out mean here.
    #x -= np.mean(x, axis=0)
    #xc -= np.mean(xc, axis=0)
    # TODO(orenlivne): implement this ourselves instead of trying to fit it into sklearn's API?
    d = sklearn.metrics.pairwise.cosine_similarity(x.transpose(), xc.transpose())
    print(d)
    return d
