import numpy as np
from hexrd.core import imageseries

_NFXY = (3, 7, 5)


class ImageSeriesTest:
    """Base class for shared test setup or methods."""
    pass


# random array from randint
random_array = np.array(
    [[[2,  4,  5,  0, 14, 16, 17],
      [18, 17,  5, 19,  2,  8, 17],
      [0, 16, 10, 18, 13, 16,  9],
      [2, 15, 13, 14, 12, 19,  9],
      [0,  3,  4, 11,  8,  8,  3]],

     [[8, 17, 15,  0,  0,  5, 17],
      [7,  4,  8, 17,  2,  5,  3],
      [14,  1, 12,  4,  6, 19,  2],
      [13,  7,  5,  6, 17,  17,  6],
      [16,  4, 10,  3,  6,  0, 14]],

     [[17,  3,  8,  3, 15,  6, 18],
      [13,  1,  3,  5,  9, 11, 15],
      [1, 11, 15,  1, 19,  2,  0],
      [5,  0, 12, 11, 12, 10, 11],
      [6,  4, 16,  2, 16,  9, 18]]]
)


def make_array_ims():
    """Returns both the array and the array imageseries."""
    is_a = imageseries.open(
        None, 'array', data=random_array, meta=make_meta()
    )
    return random_array, is_a


def make_meta():
    """Create sample metadata for testing."""
    return {'testing': np.array([1, 2, 3])}


def make_omega_meta(n):
    """Create omega metadata for testing."""
    return np.linspace((0, 0), (1, 1), n)


def compare(ims1, ims2):
    """Compare two imageseries."""
    if len(ims1) != len(ims2):
        raise ValueError("Lengths do not match")

    if ims1.dtype != ims2.dtype:
        raise ValueError(
            f"Types do not match: {repr(ims1.dtype)} is not {repr(ims2.dtype)}"
        )

    maxdiff = 0.0
    for i in range(len(ims1)):
        f1 = ims1[i]
        f2 = ims2[i]
        fdiff = np.linalg.norm(f1 - f2)
        maxdiff = np.maximum(maxdiff, fdiff)

    return maxdiff


def compare_meta(ims1, ims2):
    """Compare metadata of two imageseries."""
    t1 = ims1.metadata['testing']
    t2 = ims2.metadata['testing']

    return np.all(t1 == t2)
