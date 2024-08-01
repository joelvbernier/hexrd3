"""
Not a complete test, just checking that the stress and strain tensor vs
vector representation changes are inverses
"""

import numpy as np
from hexrd.core import matrixutil as mu


def test_stress_repr():
    """Test the representations of stress"""
    for _ in range(100):
        vec = np.random.rand(6)
        ten = mu.stress_vec_to_ten(vec)
        vec_back = mu.stress_ten_to_vec(ten).T[0]
        assert np.allclose(vec, vec_back)


def test_strain_repr():
    """Test the representations of strain"""
    for _ in range(100):
        vec = np.random.rand(6)
        ten = mu.strain_vec_to_ten(vec)
        vec_back = mu.strain_ten_to_vec(ten).T[0]
        assert np.allclose(vec, vec_back)
