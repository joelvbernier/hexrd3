import numpy as np
from hexrd.core import matrixutil as mu


def test_unit_vector(n_dim):
    """Test normalizing column vectors"""
    np.random.seed(0)
    for _ in range(100):
        if n_dim == 1:
            v = np.random.rand(10)
        else:
            v = np.random.rand(10, 10)
        v_unit_expected = v / np.linalg.norm(v, axis=0)
        v_unit = mu.unit_vector(v)
        assert np.allclose(v_unit, v_unit_expected)


def pytest_generate_tests(metafunc):
    """
    Make sure methods work on different dimension sizes.
    """
    if 'n_dim' in metafunc.fixturenames:
        metafunc.parametrize('n_dim', [1, 2])
