import os
import shutil
import tempfile
import numpy as np
import pytest
from hexrd.core import imageseries
from .common import make_array_ims, compare, compare_meta


@pytest.fixture(scope='class')
def tmpdir():
    """Fixture for temporary directory setup and teardown."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


@pytest.fixture
def h5_file(tmpdir):
    """Fixture for HDF5 test file path."""
    return os.path.join(tmpdir, 'test_ims.h5')


@pytest.fixture
def h5_path():
    """Fixture for HDF5 path."""
    return 'array-data'


@pytest.fixture
def is_a():
    """Fixture for creating array image series."""
    _, is_a = make_array_ims()
    return is_a


@pytest.fixture
def fc_file(tmpdir):
    """Fixture for frame-cache file."""
    return os.path.join(tmpdir, 'frame-cache.npz')


@pytest.fixture
def threshold():
    """Fixture for frame-cache threshold."""
    return 0.5


@pytest.fixture
def cache_file():
    """Fixture for cache file."""
    return 'frame-cache.npz'


class TestFormatH5:

    def test_fmt_h5(self, h5_file, h5_path, is_a):
        """Save/load HDF5 format."""
        imageseries.write(is_a, h5_file, 'hdf5', path=h5_path)
        is_h = imageseries.open(h5_file, 'hdf5', path=h5_path)

        diff = compare(is_a, is_h)
        assert diff == pytest.approx(0.0), "HDF5 reconstruction failed"
        assert compare_meta(is_a, is_h)

    def test_fmt_h5_np_array(self, h5_file, h5_path, is_a):
        """HDF5 format with numpy array metadata."""
        key = 'np-array'
        npa = np.array([0, 2.0, 1.3])
        is_a.metadata[key] = npa
        imageseries.write(is_a, h5_file, 'hdf5', path=h5_path)
        is_h = imageseries.open(h5_file, 'hdf5', path=h5_path)
        meta = is_h.metadata

        diff = np.linalg.norm(meta[key] - npa)
        assert diff == pytest.approx(0.0), "HDF5 numpy array metadata failed"

    def test_fmt_h5_no_compress(self, h5_file, h5_path, is_a):
        """HDF5 options: no compression."""
        imageseries.write(is_a, h5_file, 'hdf5', path=h5_path, gzip=0)
        is_h = imageseries.open(h5_file, 'hdf5', path=h5_path)

        diff = compare(is_a, is_h)
        assert diff == pytest.approx(0.0), "HDF5 reconstruction failed"
        assert compare_meta(is_a, is_h)

    def test_fmt_h5_compress_err(self, h5_file, h5_path, is_a):
        """HDF5 options: compression level out of range."""
        with pytest.raises(ValueError):
            imageseries.write(is_a, h5_file, 'hdf5', path=h5_path, gzip=10)

    def test_fmt_h5_chunk(self, h5_file, h5_path, is_a):
        """HDF5 options: chunk size."""
        imageseries.write(is_a, h5_file, 'hdf5', path=h5_path, chunk_rows=0)
        is_h = imageseries.open(h5_file, 'hdf5', path=h5_path)

        diff = compare(is_a, is_h)
        assert diff == pytest.approx(0.0), "HDF5 reconstruction failed"
        assert compare_meta(is_a, is_h)


class TestFormatFrameCache:

    def test_fmt_fc(self, fc_file, is_a, threshold, cache_file):
        """Save/load frame-cache format."""
        imageseries.write(
            is_a,
            fc_file,
            'frame-cache',
            threshold=threshold,
            cache_file=cache_file,
        )
        is_fc = imageseries.open(fc_file, 'frame-cache')
        diff = compare(is_a, is_fc)
        assert diff == pytest.approx(0.0), "Frame-cache reconstruction failed"
        assert compare_meta(is_a, is_fc)

    def test_fmt_fc_no_cache_file(self, fc_file, is_a, threshold):
        """Save/load frame-cache format with no cache_file arg."""
        imageseries.write(is_a, fc_file, 'frame-cache', threshold=threshold)
        is_fc = imageseries.open(fc_file, 'frame-cache')
        diff = compare(is_a, is_fc)
        assert diff == pytest.approx(0.0), "Frame-cache reconstruction failed"
        assert compare_meta(is_a, is_fc)

    def test_fmt_fc_np_array(self, fc_file, is_a, threshold, cache_file):
        """Frame-cache format with numpy array metadata."""
        key = 'np-array'
        npa = np.array([0, 2.0, 1.3])
        is_a.metadata[key] = npa

        imageseries.write(
            is_a,
            fc_file,
            'frame-cache',
            threshold=threshold,
            cache_file=cache_file,
        )
        is_fc = imageseries.open(fc_file, 'frame-cache')
        meta = is_fc.metadata
        diff = np.linalg.norm(meta[key] - npa)
        assert diff == pytest.approx(
            0.0
        ), "Frame-cache numpy array metadata failed"
