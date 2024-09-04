import pytest
import numpy as np
from unittest.mock import mock_open, patch
from hexrd.core.imageseries.load.rawimage import RawImageSeriesAdapter

# Sample YAML data for mocking the YAML input
sample_yaml_content = """
filename: "mock_image_file.raw"
scalar:
  type: "i"
  bytes: 4
  signed: True
  endian: "little"
shape: "100 100"
skip: 0
"""

@pytest.fixture
def mock_open_file():
    """Mock the open() call and return a file-like object."""
    with patch("builtins.open", mock_open(read_data=sample_yaml_content)) as mock_file:
        yield mock_file

@pytest.fixture
def mock_getsize():
    """Mock os.path.getsize() to return a fixed file size."""
    with patch("os.path.getsize", return_value=40000):  # Example file size
        yield

@pytest.fixture
def adapter(mock_open_file, mock_getsize):
    """Fixture to create an instance of RawImageSeriesAdapter."""
    return RawImageSeriesAdapter("mock_yaml_file.yml")

def test_initialization(adapter):
    """Test the initialization of RawImageSeriesAdapter."""
    assert isinstance(adapter, RawImageSeriesAdapter)
    assert adapter.fname == "mock_image_file.raw"
    assert adapter.shape == (100, 100)
    assert adapter.skipbytes == 0
    assert adapter._frame_bytes == 4 * 100 * 100  # dtype itemsize * frame size
    assert adapter._len == 1  # Since mock file size is 40000, there's 1 frame


def test_len(adapter):
    """Test the __len__ method."""
    assert len(adapter) == 1


def test_shape(adapter):
    """Test the shape property."""
    assert adapter.shape == (100, 100)


def test_getitem(adapter):
    """Test the __getitem__ method."""
    # Mock np.fromfile to return a simple array of the correct shape
    with patch("numpy.fromfile", return_value=np.zeros(100 * 100)):
        frame = adapter[0]
        assert frame.shape == (100, 100)
        assert np.array_equal(frame, np.zeros((100, 100)))


def test_getitem_invalid_index(adapter):
    """Test __getitem__ with an invalid index."""
    # Mock np.fromfile to simulate file reading without real files
    with patch("numpy.fromfile", return_value=np.zeros(100 * 100)):
        with pytest.raises(IndexError):
            adapter[5]


def test_get_dtype(adapter):
    """Test the _get_dtype method."""
    expected_dtype = np.dtype("<i4")  # Little endian signed int with 4 bytes
    assert adapter.dtype == expected_dtype


def test_typechars():
    """Test the static method typechars."""
    # Test for 4-byte signed little-endian integer
    assert RawImageSeriesAdapter.typechars("i", 4, True, True) == "<i"
    # Test for 4-byte unsigned big-endian integer
    assert RawImageSeriesAdapter.typechars("i", 4, False, False) == ">I"
    # Test for float
    assert RawImageSeriesAdapter.typechars("f", 4, little=True) == "<f"
    # Test for double
    assert RawImageSeriesAdapter.typechars("d", 8, little=False) == ">d"
