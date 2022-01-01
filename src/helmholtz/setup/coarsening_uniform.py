"""coarsening (R) construction routines. Based on SVD on an aggregate."""
import logging
import numpy as np
import pandas as pd
import scipy.sparse
from numpy.linalg import svd
from typing import List, Tuple

import helmholtz as hm

_LOGGER = logging.getLogger(__name__)


class FixedAggSizeUniformCoarsener:
    """
    Creates the next coarse level's SVD coarsening operator R on a full domain (non-repetitive/repetitive).
    Uses a specified fixed-size aggregate and #PCs throughout the domain.
    """
    def __init__(self, x, aggregate_size, cycle_index: float = 1,
                 cycle_coarse_level_work_bound: float = 0.7, repetitive: bool = False):
        """
        Creates an object that creates coarsening candidates with different # components and fixed aggregate size.

        Args:
            x: fine-level test matrix.
            cycle_index: cycle index of the cycle we are designing.
            cycle_coarse_level_work_bound: cycle_index * max_coarsening_ratio. Bounds the proportion of coarse level
                cycle work.
            aggregate_size: uniform aggregate size throughout the domain.
            repetitive: whether to exploit problem repetitiveness by creating a constant R stencil on all aggregates
                using windows from a single (or few) test vectors.

        Returns: a generator of (coarsening operator R, mean energy error over all aggregates), for all
            num_components = 1..max_components (that satisfy the cycle work bound).
        """
        assert (0 <= cycle_coarse_level_work_bound <= 1) and (cycle_index > 0)
        self._x = x
        self._domain_size = x.shape[0]
        self._aggregate_size = aggregate_size
        self._cycle_index = cycle_index
        self._max_coarsening_ratio = cycle_coarse_level_work_bound
        self._repetitive = repetitive
        self._starts = hm.linalg.get_uniform_aggregate_starts(self._domain_size, aggregate_size)
        self._r_aggregate_candidates, self._s_aggregate = self._candidate_coarsening()

    @property
    def s_aggregate(self):
        """Returns the singular values of each aggregate."""
        return self._s_aggregate

    @property
    def smallest_singular_value_loss(self):
        s2 = self._s_aggregate ** 2
        return (np.sum(s2[-1]) / np.sum(s2[0])) ** 0.5

    @property
    def mean_energy_error(self):
        return np.mean(
            np.clip((1 - np.cumsum(self._s_aggregate ** 2, axis=1) / np.sum(self._s_aggregate ** 2, axis=1)[:, None]),
                    0, None) ** 0.5, axis=0)

    @property
    def _num_components(self):
        """Returns the maximum number of components to keep in each aggregate SVD."""
        return int(self._aggregate_size * self._max_coarsening_ratio / self._cycle_index)

    def __getitem__(self, nc) -> Tuple[scipy.sparse.csr_matrix, List[np.ndarray]]:
        """
        Creates the next coarse level's SVD coarsening operator R on a full domain (non-repetitive).
        Uses a fixed-size aggregate.

        Args:
            nc = number of principal components to keep per aggregate.

        Returns: a generator of (coarsening operator R, mean energy error over all aggregates), for all
            nc = 1..max_components (that satisfy the cycle work bound).
        """
        # Merge all aggregate coarsening operators into R.
        r_aggregate_candidates, s_aggregate = self._r_aggregate_candidates, self._s_aggregate
        r_aggregate = [candidate[:nc] for candidate in r_aggregate_candidates]
        if self._domain_size % self._aggregate_size == 0:
            # Non-overlapping aggregates => R is block diagonal.
            r = scipy.sparse.block_diag(r_aggregate).tocsr()
        else:
            # Overlapping aggregate. Form the block-diagonal matrix except the last aggregate, then add it in.
            r = scipy.sparse.block_diag(r_aggregate[:-1]).tocsr()
            # Add columns to the matrix of the "interior" aggregates.
            r = scipy.sparse.csr_matrix((r.data, r.indices, r.indptr), (r.shape[0], self._domain_size))
            # Create a matrix for the boundary aggregate.
            r_last = scipy.sparse.csr_matrix(r_aggregate[-1])
            offset = self._starts[-1]
            r_last = scipy.sparse.csr_matrix((r_last.data, r_last.indices + offset, r_last.indptr),
                                             shape=(r_last.shape[0], self._domain_size)).todense()
            # Merge the two.
            r = scipy.sparse.vstack((r, r_last)).tocsr()
        return r, self.mean_energy_error[nc - 1]

    def all_candidate_coarsening(self):
        """Returns a list of all candidate coarsening for all nc = 1..max_components."""
        return [(nc, self[nc]) for nc in range(1, self._num_components + 1)]

    def _candidate_coarsening(self):
        """Sweeps the domain left to right; add an aggregate and its coarsening until we get to the domain end.
        Uses a uniform aggregate size; the last two aggregates will overlap of the domain size is not divisible by the
        aggregate size. Returns PCs of each aggregate and singular values of each aggregate."""
        x = self._x
        if self._repetitive:
            # Keep enough windows=samples (4 * aggregate_size) an over-determined LS problem for R.
            x_aggregate_t = hm.linalg.get_windows_by_index(
                x, np.arange(self._aggregate_size), 1, 4 * self._aggregate_size)
            # Tile the same coarsening over all aggregates.
            aggregate_coarsening = create_coarsening(x_aggregate_t, self._num_components)
            svd_results = [aggregate_coarsening for _ in self._starts]
        else:
            svd_results = [create_coarsening(x[start:start + self._aggregate_size].T, self._num_components)
                           for start in self._starts]
        r_aggregate_candidates = tuple(aggregate_svd_result[0] for aggregate_svd_result in svd_results)
        # Singular values, used for checking energy error in addition to the mock cycle criterion.
        s_aggregate = np.concatenate(tuple(aggregate_svd_result[1][None, :] for aggregate_svd_result in svd_results))
        return r_aggregate_candidates, s_aggregate


class UniformCoarsener:
    """
    Creates the next coarse level's SVD coarsening operator R on a full domain (non-repetitive).
    Uses a fixed-size aggregate and #PCs throughout the domain. Automatically determines the optimal aggregate size.
    """
    def __init__(self, level, x, nu: int,
                 max_aggregate_size: int = 8,
                 cycle_index: float = 1,
                 cycle_coarse_level_work_bound: float = 0.7,
                 max_energy_loss: float = 0.2,
                 expected_eventual_coarsening_ratio: float = 0.5,
                 min_trusted_mock_conv_factor: float = 0.05,
                 efficiency_leeway_factor: float = 1.05,
                 repetitive: bool = False):
        """
        Args:
            level: level object containing the relaxation scheme.
            x: fine-level test matrix.
            max_aggregate_size: maximum aggregate size to consider.
            nu: #relation sweeps per mock cycle.
            cycle_index: cycle index of the cycle we are designing.
            cycle_coarse_level_work_bound: cycle_index * max_coarsening_ratio. Bounds the proportion of coarse level
                work in the cycle.
            max_energy_loss: determines the minimum aggregate size considered, such that
                sigma_min < max_energy_loss * sigma_max (or sqrt of ratio of the corresponding sum of squares over all
                aggregates, in the non-repetitive case).
            expected_eventual_coarsening_ratio: coarsening ratio of subsequent coarsening to assume in estimating the
                cycle work.
            min_trusted_mock_conv_factor: the minimum mock cycle convergence factor value that's "trusted". Any value
                smaller than this value will be set to 'min_trusted_mock_conv_factor' since it's "too good" in reality
                due to interpolation errors.
            efficiency_leeway_factor: consider all cases with efficiency measure < efficiency_leeway_factor *
                min(measure) as candidates. Pick the one with the smallest aggregate size.
            repetitive: whether to exploit problem repetitiveness by creating a constant R stencil on all aggregates
                using windows from a single (or few) test vectors.
        """
        # Generates coarse variables (R) on the non-repetitive domain.
        self._cycle_index = cycle_index
        self._expected_eventual_coarsening_ratio = expected_eventual_coarsening_ratio
        self._efficiency_leeway_factor = efficiency_leeway_factor
        self._min_trusted_mock_conv_factor = min_trusted_mock_conv_factor

        domain_size = x.shape[0]
        aggregate_size_values = np.arange(2, max_aggregate_size + 1, dtype=int)
        if repetitive:
            # In a repetitive framework, ensure that the aggregate size divides the domain size.
            aggregate_size_values = np.array([a for a in aggregate_size_values if domain_size % a == 0])
        _LOGGER.debug("aggregate_size_values {}".format(np.array2string(aggregate_size_values, separator=", ")))
        agg_coarsener = [FixedAggSizeUniformCoarsener(
            x, aggregate_size, cycle_index=cycle_index,
            cycle_coarse_level_work_bound=cycle_coarse_level_work_bound,
            repetitive=repetitive)
            for aggregate_size in aggregate_size_values]
        # Only consider aggregate size such that the minimum singular value < smallest_singular_value_loss * max
        # singular value (averaged over domain).
        smallest_singular_value_loss = np.array([coarsener.smallest_singular_value_loss for coarsener in agg_coarsener])
        min_aggregate_size_index = np.argmin(smallest_singular_value_loss < max_energy_loss)
        _LOGGER.debug("smallest_singular_value_loss {} min size index {} min agg size {}".format(
            np.array2string(smallest_singular_value_loss, separator=", ", precision=2),
            min_aggregate_size_index, aggregate_size_values[min_aggregate_size_index]))
        aggregate_size_values = aggregate_size_values[min_aggregate_size_index:]
        agg_coarsener = agg_coarsener[min_aggregate_size_index:]

        self._result = [(aggregate_size, coarsening[0], coarsening[1][0], coarsening[1][1])
                        for aggregate_size, coarsener in zip(aggregate_size_values, agg_coarsener)
                        for coarsening in coarsener.all_candidate_coarsening()]
        self._nu = nu
        r_values = np.array([item[2] for item in self._result])
        self._mock_conv_factor = np.array([hm.setup.auto_setup.mock_cycle_conv_factor(level, r, nu) for r in r_values])

    # TODO(oren): max_conv_factor can be derived from cycle index instead of being passed in.
    def get_coarsening_info(self, max_conv_factor, fmt: str = "array"):
        """
        Returns a table of coarsening matrix performance statistics vs. aggregate size and # principal components
        (coarse vars per aggregate).

        Args:
            max_conv_factor: max convergence factor to allow. NOTE: in principle, should be derived from cycle index.

        Returns:
            table of index into the _result array, aggregate_size, nc, cr, mean_energy_error, mock_conv, mock_work,
            mock_efficiency.
        """
        aggregate_size = np.array([item[0] for item in self._result])
        nc = np.array([item[1] for item in self._result])
        r_values = np.array([item[2] for item in self._result])
        mean_energy_error = np.array([item[3] for item in self._result])
        # Coarsening ratio (first coarsening).
        cr = np.array([r.shape[0] / r.shape[1] for r in r_values])
        # Coarsening ratio (subsequent coarsening). Assuming that it can't be better than
        # # self._expected_eventual_coarsening_ratio.
        cr_subsequent = np.clip(cr, self._expected_eventual_coarsening_ratio, None)
        work = self._nu * (self._cycle_index * (cr - cr_subsequent) + 1 / (1 - self._cycle_index * cr_subsequent))
        efficiency = np.clip(self._mock_conv_factor, self._min_trusted_mock_conv_factor, None) ** (1 / work)
        candidate = self._mock_conv_factor <= max_conv_factor
        i = np.where(candidate)[0]
        candidate = np.vstack((
            i,
            aggregate_size[i],
            nc[i],
            cr[i],
            mean_energy_error[i],
            self._mock_conv_factor[candidate],
            work[candidate],
            efficiency[candidate]
        )).T
        if fmt == "dataframe":
            columns = ("i", "a", "nc", "cr", "Energy Error", "conv", "work", "eff")
            candidate = pd.DataFrame(candidate).rename(columns=dict(enumerate(columns))).astype(
                {"i": "int32", "a": "int32", "nc": "int32"})
        return candidate
#        return np.array(candidate, [("i", "i4"), ("a", "i4"), ("nc", "i4"), ("cr", "f8"), ("Energy Error", "f8"),
#                                    ("conv", "f8"), ("work", "f8"), ("eff", "f8")])

    # TODO(oren): max_conv_factor can be derived from cycle index instead of being passed in.
    def get_optimal_coarsening(self, max_conv_factor, aggregate_size: int = None):
        """
        Returns a coarsening matrix (R) on the non-repetitive domain, which maximizes mock cycle efficiency over
        aggregate size and # principal components (coarse vars per aggregate).

        Args:
            max_conv_factor: max convergence factor to allow. NOTE: in principle, should be derived from cycle index.

        Returns:
            Optimal R, aggregate_size, nc, cr, mean_energy_error, nu, mock_conv, mock_work, mock_efficiency.
        """
        candidate = self.get_coarsening_info(max_conv_factor)
        if aggregate_size is not None:
            candidate = candidate[candidate[:, 1] == aggregate_size]
        if candidate.size == 0:
            _LOGGER.info("Candidates coarsening")
            _LOGGER.info(self.get_coarsening_info(1.0))
            raise Exception("Could not find a coarsening whose mock cycle is below {:.2f}".format(max_conv_factor))
        # Find all cases that are within 5% of the best efficiency.
        eff = candidate[:, -1]
        candidate = candidate[eff < self._efficiency_leeway_factor * min(eff)]
        # Of those, pick the minimal aggregate size.
        a = candidate[:, 1].astype(int)
        aggregate_size = min(a)
        candidate = candidate[a == aggregate_size]
        # Of those, pick the one with the best efficiency.
        best_index = np.argmin(candidate[:, -1])
        i, aggregate_size, nc, cr, mean_energy_error, mock_conv, mock_work, mock_efficiency = candidate[best_index]
        return self._result[int(i)][2], int(aggregate_size), int(nc), cr, mean_energy_error, mock_conv, \
            mock_work, mock_efficiency


def create_coarsening(x_aggregate_t: np.ndarray, num_components: int, normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates R (coarse variables) on an aggregate from SVD principal components.

    Args:
        x_aggregate_t: fine-level test matrix on an aggregate, transposed.
        num_components: number of principal components.
        normalize: if True, scales the row sums of R to 1.

    Returns:
        coarsening matrix nc x {aggregate_size} (dense), list of ALL singular values on aggregate.
    """
    u, s, vh = svd(x_aggregate_t)
    r = vh[:num_components]
    if normalize:
        r /= r.sum(axis=1)[:, None]
    return r, s
