import pytest
import numpy as np
from hexrd.core.imageseries.load.array import ArrayImageSeriesAdapter


@pytest.fixture
def sample_3d_data():
    """Fixture for a sample 3D numpy array."""
    return np.random.rand(5, 10, 10)  # 5 frames, each 10x10


@pytest.fixture
def sample_2d_data():
    """Fixture for a sample 2D numpy array."""
    return np.random.rand(10, 10)  # A single 10x10 image


@pytest.fixture
def sample_metadata():
    """Fixture for sample metadata."""
    return {'description': 'test metadata'}


def test_constructor_with_3d_array(sample_3d_data, sample_metadata):
    """Test constructor with a valid 3D array."""
    adapter = ArrayImageSeriesAdapter(
        None, data=sample_3d_data, meta=sample_metadata
    )
    assert adapter._nframes == 5
    assert adapter.shape == (10, 10)
    assert adapter.metadata == sample_metadata
    assert adapter.dtype == sample_3d_data.dtype


def test_constructor_with_2d_array(sample_2d_data):
    """Test constructor with a 2D array, which should be tiled."""
    adapter = ArrayImageSeriesAdapter(None, data=sample_2d_data)
    assert adapter._nframes == 1
    assert adapter.shape == (10, 10)


def test_invalid_ndim_array():
    """Test that a ValueError is raised with arrays of more than 3 dimensions."""
    data_4d = np.random.rand(5, 10, 10, 3)  # 4D array (invalid)
    with pytest.raises(ValueError, match='input array must be 2-d or 3-d'):
        ArrayImageSeriesAdapter(None, data=data_4d)


def test_get_item(sample_3d_data):
    """Test the __getitem__ method for correct frame indexing."""
    adapter = ArrayImageSeriesAdapter(None, data=sample_3d_data)
    np.testing.assert_array_equal(
        adapter[0], sample_3d_data[0]
    )  # Test just the first frame


def test_iterator(sample_3d_data):
    """Test that the iterator returns the correct sequence of images."""
    adapter = ArrayImageSeriesAdapter(None, data=sample_3d_data)
    for idx, frame in enumerate(adapter):
        np.testing.assert_array_equal(frame, sample_3d_data[idx])


def test_length(sample_3d_data):
    """Test that __len__ returns the correct number of frames."""
    adapter = ArrayImageSeriesAdapter(None, data=sample_3d_data)
    assert len(adapter) == 5
