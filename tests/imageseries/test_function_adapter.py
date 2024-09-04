import numpy as np
import pytest
from hexrd.core.imageseries.load.function import FunctionImageSeriesAdapter


def mock_function(index):
    """A mock function that returns a 2D numpy array based on the provided index."""
    np.random.seed(index)
    return np.random.rand(10, 10)


@pytest.fixture
def adapter_with_mock_function():
    """Fixture to create a FunctionImageSeriesAdapter with a mock function."""
    return FunctionImageSeriesAdapter(
        fname=None,
        func=mock_function,
        num_frames=5,
        meta={"description": "Test metadata"}
    )


def test_init(adapter_with_mock_function):
    """Test that the adapter initializes correctly."""
    assert adapter_with_mock_function._nframes == 5
    assert adapter_with_mock_function.shape == (10, 10)
    assert adapter_with_mock_function.dtype == np.float64
    assert adapter_with_mock_function.metadata["description"] == "Test metadata"


def test_getitem(adapter_with_mock_function):
    """Test retrieving frames by index."""
    frame = adapter_with_mock_function[0]
    assert frame.shape == (10, 10)
    assert frame.dtype == np.float64

    # Ensure that different indices give different frames
    frame_1 = adapter_with_mock_function[1]
    frame_2 = adapter_with_mock_function[2]
    assert not np.array_equal(frame_1, frame_2)

    # Test for an invalid key
    with pytest.raises(TypeError) as excinfo:
        adapter_with_mock_function["invalid_key"]
    assert str(excinfo.value) == "Key must be an integer, but received str."


def test_len(adapter_with_mock_function):
    """Test that the length of the adapter is correct."""
    assert len(adapter_with_mock_function) == 5


def test_iter(adapter_with_mock_function):
    """Test iterating over the frames."""
    frames = list(adapter_with_mock_function)
    assert len(frames) == 5
    assert all(frame.shape == (10, 10) for frame in frames)


def test_metadata_property(adapter_with_mock_function):
    """Test accessing the metadata."""
    metadata = adapter_with_mock_function.metadata
    assert isinstance(metadata, dict)
    assert metadata["description"] == "Test metadata"


def test_shape_property(adapter_with_mock_function):
    """Test accessing the shape property."""
    assert adapter_with_mock_function.shape == (10, 10)


def test_dtype_property(adapter_with_mock_function):
    """Test accessing the dtype property."""
    assert adapter_with_mock_function.dtype == np.float64


def test_boundary_conditions(adapter_with_mock_function):
    """Test accessing the first and last frames."""
    first_frame = adapter_with_mock_function[0]
    last_frame = adapter_with_mock_function[4]
    assert first_frame.shape == (10, 10)
    assert last_frame.shape == (10, 10)

    with pytest.raises(IndexError):
        adapter_with_mock_function[5]
