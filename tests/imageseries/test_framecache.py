import os
import numpy as np
import pytest
from unittest import mock
from scipy.sparse import csr_matrix
from hexrd.core.imageseries.load.framecache import FrameCacheImageSeriesAdapter


@pytest.fixture
def mock_npz_file():
    """Mock NPZ file content."""
    return {
        'nframes': np.array([3]),
        'shape': np.array([5, 5]),
        'dtype': 'float32',
        '0_row': np.array([0, 1, 2]),
        '0_col': np.array([0, 1, 2]),
        '0_data': np.array([1, 2, 3]),
        '1_row': np.array([0, 1, 2]),
        '1_col': np.array([0, 1, 2]),
        '1_data': np.array([4, 5, 6]),
        '2_row': np.array([0, 1, 2]),
        '2_col': np.array([0, 1, 2]),
        '2_data': np.array([7, 8, 9])
    }


@pytest.fixture
def mock_yaml_file():
    """Mock YAML file content."""
    return {
        'data': {
            'file': 'mock_cache_file.npz',
            'nframes': 3,
            'shape': [5, 5],
            'dtype': 'float32'
        },
        'meta': {
            'description': 'mock metadata'
        }
    }


@pytest.fixture
def adapter_npz(mock_npz_file):
    """Fixture to create an adapter with NPZ data."""
    with mock.patch('numpy.load', return_value=mock_npz_file):
        return FrameCacheImageSeriesAdapter('mock_cache_file.npz')


@pytest.fixture
def adapter_yaml(mock_npz_file, mock_yaml_file):
    """Fixture to create an adapter with YAML data."""
    with mock.patch('yaml.safe_load', return_value=mock_yaml_file):
        with mock.patch('numpy.load', return_value=mock_npz_file):
            with mock.patch('builtins.open', mock.mock_open(read_data="mock file content")):
                return FrameCacheImageSeriesAdapter('mock_file.yml', style='yml')


def test_init_npz(adapter_npz):
    """Test initialization with NPZ file."""
    assert len(adapter_npz) == 3
    assert adapter_npz._shape == (5, 5)
    assert adapter_npz._dtype == np.float32
    assert isinstance(adapter_npz.metadata, dict)


def test_init_yaml(adapter_yaml):
    """Test initialization with YAML file."""
    assert len(adapter_yaml) == 3
    assert adapter_yaml._shape == (5, 5)
    assert adapter_yaml._dtype == np.float32
    assert adapter_yaml.metadata['description'] == 'mock metadata'


def test_metadata_property(adapter_yaml):
    """Test the metadata property."""
    assert 'description' in adapter_yaml.metadata
    assert adapter_yaml.metadata['description'] == 'mock metadata'


def test_dtype_property(adapter_npz):
    """Test the dtype property."""
    assert adapter_npz.dtype == np.float32


def test_shape_property(adapter_npz):
    """Test the shape property."""
    assert adapter_npz.shape == (5, 5)


def test_len_method(adapter_npz):
    """Test the __len__ method."""
    assert len(adapter_npz) == 3


def test_getitem_method(adapter_npz, mock_npz_file):
    """Test the __getitem__ method."""
    frame_0 = adapter_npz[0]
    expected_frame = csr_matrix((mock_npz_file['0_data'],
                                 (mock_npz_file['0_row'],
                                  mock_npz_file['0_col'])),
                                shape=(5, 5)).toarray()
    assert np.array_equal(frame_0, expected_frame)


def test_iter_method(adapter_npz):
    """Test the __iter__ method."""
    frames = list(adapter_npz)
    assert len(frames) == 3
    assert isinstance(frames[0], np.ndarray)
    assert frames[0].shape == (5, 5)


def test_load_cache_npz(mock_npz_file):
    """Test loading frames from NPZ."""
    with mock.patch('numpy.load', return_value=mock_npz_file):
        adapter = FrameCacheImageSeriesAdapter('mock_cache_file.npz')
        assert len(adapter) == 3
        frame = adapter[0]
        assert frame.shape == (5, 5)


def test_load_cache_yaml(mock_npz_file, mock_yaml_file):
    """Test loading frames from YAML."""
    with mock.patch('yaml.safe_load', return_value=mock_yaml_file):
        with mock.patch('numpy.load', return_value=mock_npz_file):
            with mock.patch('builtins.open', mock.mock_open(read_data="mock file content")):
                adapter = FrameCacheImageSeriesAdapter('mock_file.yml', style='yml')
                assert len(adapter) == 3
                frame = adapter[0]
                assert frame.shape == (5, 5)
