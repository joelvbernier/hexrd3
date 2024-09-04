import pytest
from .common import make_array_ims


@pytest.fixture
def array_and_imageseries():
    """Fixture to create the array and imageseries."""
    return make_array_ims()


def test_prop_nframes(array_and_imageseries):
    """Test that the number of frames matches the length of the imageseries."""
    a, is_a = array_and_imageseries
    assert a.shape[0] == len(is_a), "Number of frames does not match"


def test_prop_shape(array_and_imageseries):
    """Test that the shape of the array matches the imageseries shape."""
    a, is_a = array_and_imageseries
    assert (
        a.shape[1:] == is_a.shape
    ), "Shape mismatch between array and imageseries"


def test_prop_dtype(array_and_imageseries):
    """Test that the dtype of the array matches the imageseries dtype."""
    a, is_a = array_and_imageseries
    assert (
        a.dtype == is_a.dtype
    ), "Dtype mismatch between array and imageseries"
