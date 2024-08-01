import numpy as np

from hexrd.core import matrixutil as mutil


def test_vec_mv_cob_matrix():
    """
    Test the vec_mv_cob_matrix method with random testcases against old
    implementation
    """
    np.random.seed(0)
    # Generate some random matrices
    r = np.random.rand(20, 3, 3) * 2 - 1

    t = np.zeros((len(r), 6, 6), dtype='float64')
    sqr2 = np.sqrt(2)
    # Hardcoded implementation
    t[:, 0, 0] = r[:, 0, 0]**2
    t[:, 0, 1] = r[:, 0, 1]**2
    t[:, 0, 2] = r[:, 0, 2]**2
    t[:, 0, 3] = sqr2 * r[:, 0, 1] * r[:, 0, 2]
    t[:, 0, 4] = sqr2 * r[:, 0, 0] * r[:, 0, 2]
    t[:, 0, 5] = sqr2 * r[:, 0, 0] * r[:, 0, 1]
    t[:, 1, 0] = r[:, 1, 0]**2
    t[:, 1, 1] = r[:, 1, 1]**2
    t[:, 1, 2] = r[:, 1, 2]**2
    t[:, 1, 3] = sqr2 * r[:, 1, 1] * r[:, 1, 2]
    t[:, 1, 4] = sqr2 * r[:, 1, 0] * r[:, 1, 2]
    t[:, 1, 5] = sqr2 * r[:, 1, 0] * r[:, 1, 1]
    t[:, 2, 0] = r[:, 2, 0]**2
    t[:, 2, 1] = r[:, 2, 1]**2
    t[:, 2, 2] = r[:, 2, 2]**2
    t[:, 2, 3] = sqr2 * r[:, 2, 1] * r[:, 2, 2]
    t[:, 2, 4] = sqr2 * r[:, 2, 0] * r[:, 2, 2]
    t[:, 2, 5] = sqr2 * r[:, 2, 0] * r[:, 2, 1]
    t[:, 3, 0] = sqr2 * r[:, 1, 0] * r[:, 2, 0]
    t[:, 3, 1] = sqr2 * r[:, 1, 1] * r[:, 2, 1]
    t[:, 3, 2] = sqr2 * r[:, 1, 2] * r[:, 2, 2]
    t[:, 3, 3] = r[:, 1, 2] * r[:, 2, 1] + r[:, 1, 1] * r[:, 2, 2]
    t[:, 3, 4] = r[:, 1, 2] * r[:, 2, 0] + r[:, 1, 0] * r[:, 2, 2]
    t[:, 3, 5] = r[:, 1, 1] * r[:, 2, 0] + r[:, 1, 0] * r[:, 2, 1]
    t[:, 4, 0] = sqr2 * r[:, 0, 0] * r[:, 2, 0]
    t[:, 4, 1] = sqr2 * r[:, 0, 1] * r[:, 2, 1]
    t[:, 4, 2] = sqr2 * r[:, 0, 2] * r[:, 2, 2]
    t[:, 4, 3] = r[:, 0, 2] * r[:, 2, 1] + r[:, 0, 1] * r[:, 2, 2]
    t[:, 4, 4] = r[:, 0, 2] * r[:, 2, 0] + r[:, 0, 0] * r[:, 2, 2]
    t[:, 4, 5] = r[:, 0, 1] * r[:, 2, 0] + r[:, 0, 0] * r[:, 2, 1]
    t[:, 5, 0] = sqr2 * r[:, 0, 0] * r[:, 1, 0]
    t[:, 5, 1] = sqr2 * r[:, 0, 1] * r[:, 1, 1]
    t[:, 5, 2] = sqr2 * r[:, 0, 2] * r[:, 1, 2]
    t[:, 5, 3] = r[:, 0, 2] * r[:, 1, 1] + r[:, 0, 1] * r[:, 1, 2]
    t[:, 5, 4] = r[:, 0, 0] * r[:, 1, 2] + r[:, 0, 2] * r[:, 1, 0]
    t[:, 5, 5] = r[:, 0, 1] * r[:, 1, 0] + r[:, 0, 0] * r[:, 1, 1]

    t2 = mutil.vec_mv_cob_matrix(r)

    assert np.allclose(t, t2)
