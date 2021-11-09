import numpy as np

def interval_norm(x: np.ndarray, window_size: int) -> np.ndarray:
    """
    Returns a matrix of local L2 norms calculated over periodic B.C. windows centered around each gridpoint.
    For a matrix of size m x n, returns a matrix of size m x n whose (i, j) element is

    norm([x[j % sz] for j in range(i - (window_size + 1)// 2 + 1, i - (window_size + 1)// 2 + 1 + window_size)])

    That is, the norm is calculated along axis=0.

    :param x:
    :param window_size: size of window to calculate the local norm over.
    :return: window norm matrix.
    """
    offset = (window_size + 1) // 2 - 1
    sz = x.shape[0]
    z = np.concatenate((x[-offset:], x, x[:window_size - offset - 1]))
    indices = np.array(list(zip(range(sz), range(window_size, sz + window_size)))).flatten()
    return ((np.add.reduceat(z ** 2, indices[:-1])[::2]) / window_size) ** 0.5

    #### Unit test
    #
    # @pytest.mark.parametrize("sz,m", [(10, 3), (10, 4), (11, 4)])
    # def test_interval_norm(self, sz, m):
    #     x = 2 * np.random.random((sz, 5)) - 1
    #
    #     actual = hm.linalg.interval_norm(x, m)
    #
    #     expected = np.array(
    #         list(map(
    #             lambda x: norm(x, axis=0) / x.shape[0] ** 0.5,
    #             (np.array([x[j % sz] for j in range(i - (m + 1) // 2 + 1, i - (m + 1) // 2 + 1 + m)])
    #              for i in range(sz)))))
    #     assert_array_almost_equal(actual, expected, 10)
