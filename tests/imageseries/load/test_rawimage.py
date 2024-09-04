import os
import pytest
import numpy as np
from unittest.mock import mock_open, patch
from concurrent.futures import ThreadPoolExecutor
from hexrd.core.imageseries.load.rawimage import RawImageSeriesAdapter

# Sample YAML data for mocking the YAML input
sample_yaml_little = """
filename: "mock_image_file.raw"
scalar:
  type: "i"
  bytes: 4
  signed: True
  endian: "little"
shape: "100 100"
skip: 0
"""

sample_yaml_big = """
filename: "mock_image_file_big.raw"
scalar:
  type: "i"
  bytes: 4
  signed: True
  endian: "big"
shape: "100 100"
skip: 0
"""

@pytest.fixture
def mock_open_file():
    """Mock the open() call."""
    with patch("builtins.open", mock_open(read_data=sample_yaml_little)) as mock_file:
        yield mock_file

@pytest.fixture
def mock_getsize():
    """Mock os.path.getsize() to return a fixed file size."""
    with patch("os.path.getsize", return_value=40000):
        yield

@pytest.fixture
def adapter(mock_open_file, mock_getsize):
    return RawImageSeriesAdapter("mock_yaml_file.yml")

def test_initialization(adapter):
    assert adapter.fname == "mock_image_file.raw"
    assert adapter.shape == (100, 100)
    assert adapter._frame_bytes == 40000
    assert adapter._len == 1

def test_len(adapter):
    assert len(adapter) == 1

def test_shape(adapter):
    assert adapter.shape == (100, 100)

def test_getitem(adapter):
    with patch("numpy.fromfile", return_value=np.zeros(100 * 100)):
        frame = adapter[0]
        assert frame.shape == (100, 100)

def test_getitem_invalid_index(adapter):
    with patch("numpy.fromfile", return_value=np.zeros(100 * 100)):
        with pytest.raises(IndexError):
            adapter[5]

def test_get_dtype(adapter):
    assert adapter.dtype == np.dtype("<i4")

def test_typechars():
    assert RawImageSeriesAdapter.typechars("i", 4, True, True) == "<i"
    assert RawImageSeriesAdapter.typechars("i", 4, False, False) == ">I"
    assert RawImageSeriesAdapter.typechars("f", 4, little=True) == "<f"
    assert RawImageSeriesAdapter.typechars("d", 8, little=False) == ">d"
    assert RawImageSeriesAdapter.typechars("b", little=True) == "<?"

def test_get_dtype_invalid_endian(adapter):
    with pytest.raises(ValueError, match='endian must be "big" for "little"'):
        adapter._get_dtype({'type': 'i', 'bytes': 4, 'signed': True, 'endian': 'invalid'})

def test_get_length_invalid_size(adapter, monkeypatch):
    monkeypatch.setattr(os.path, "getsize", lambda _: adapter._frame_bytes + 1)
    with pytest.raises(ValueError, match='Total number of bytes'):
        adapter._get_length()

def test_thread_safety(adapter):
    with patch("numpy.fromfile", return_value=np.zeros(100 * 100)):
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda idx: adapter[idx], [0, 0]))
        assert all(np.array_equal(res, np.zeros((100, 100))) for res in results)

def test_metadata_property(adapter):
    assert isinstance(adapter.metadata, dict)

def test_iter(adapter):
    with patch("numpy.fromfile", return_value=np.zeros(100 * 100)):
        for frame in adapter:
            assert frame.shape == (100, 100)

def test_big_endian(mock_open_file, mock_getsize):
    with patch("builtins.open", mock_open(read_data=sample_yaml_big)):
        adapter = RawImageSeriesAdapter("mock_yaml_file.yml")
        assert adapter.fname == "mock_image_file_big.raw"
        assert adapter.dtype == np.dtype(">i4")
