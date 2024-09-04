import warnings
import pytest
import h5py
import numpy as np
from hexrd.core.imageseries.load.hdf5 import HDF5ImageSeriesAdapter
from hexrd.core.imageseries.imageseriesiter import ImageSeriesIterator


@pytest.fixture
def hdf5_file(tmp_path):
    fname = tmp_path / 'test.h5'
    with h5py.File(fname, 'w') as f:
        grp = f.create_group('images')
        data = np.random.rand(10, 10, 10)
        grp.create_dataset('images', data=data)
        for k in ['meta1', 'meta2']:
            grp.attrs[k] = f'{k}_value'
    return str(fname)


def test_initialization_with_file_path(hdf5_file):
    adapter = HDF5ImageSeriesAdapter(hdf5_file, path='images')
    assert adapter._h5file is not None
    assert adapter._images == 'images/images'


def test_initialization_with_h5py_file(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        adapter = HDF5ImageSeriesAdapter(f, path='images')
        assert adapter._h5file == f


def test_close(hdf5_file):
    adapter = HDF5ImageSeriesAdapter(hdf5_file, path='images')
    adapter.close()
    assert adapter._h5file is None


def test_getitem(hdf5_file):
    adapter = HDF5ImageSeriesAdapter(hdf5_file, path='images')
    assert np.array_equal(adapter[0], adapter._h5file['images/images'][0])


def test_len(hdf5_file):
    adapter = HDF5ImageSeriesAdapter(hdf5_file, path='images')
    assert len(adapter) == 10


def test_pickle_state(hdf5_file):
    adapter = HDF5ImageSeriesAdapter(hdf5_file, path='images')
    state = adapter.__getstate__()

    assert '_h5file' not in state
    assert '_image_dataset' not in state
    assert '_data_group' not in state

    new_adapter = HDF5ImageSeriesAdapter(hdf5_file, path='images')
    new_adapter.__setstate__(state)

    assert new_adapter._images == 'images/images'
    assert new_adapter._h5file is not None
    assert isinstance(new_adapter._h5file, h5py.File)


def test_metadata_property(hdf5_file):
    adapter = HDF5ImageSeriesAdapter(hdf5_file, path='images')
    assert adapter.metadata['meta1'] == 'meta1_value'
    assert adapter.metadata['meta2'] == 'meta2_value'


def test_dtype_property(hdf5_file):
    adapter = HDF5ImageSeriesAdapter(hdf5_file, path='images')
    assert adapter.dtype == np.float64


def test_shape_property(hdf5_file):
    adapter = HDF5ImageSeriesAdapter(hdf5_file, path='images')
    assert adapter.shape == (10, 10)


def test_load_data_invalid_ndim(hdf5_file):
    with h5py.File(hdf5_file, 'a') as f:
        del f['images/images']
        f.create_dataset('images/images', data=np.random.rand(10, 10, 10, 10))

    with pytest.raises(
        RuntimeError,
        match='Image data must be a 2-d or 3-d array; yours is 4',
    ):
        HDF5ImageSeriesAdapter(hdf5_file, path='images')


def test_del_method(hdf5_file):
    adapter = HDF5ImageSeriesAdapter(hdf5_file, path='images')
    adapter.close()

    with warnings.catch_warnings(record=True) as w:
        del adapter
        assert len(w) == 1
        assert 'HDF5ImageSeries could not close h5 file' in str(w[-1].message)


def test_getitem_indexerror(hdf5_file):
    with h5py.File(hdf5_file, 'a') as f:
        del f['images/images']
        f.create_dataset('images/images', data=np.random.rand(10, 10))

    adapter = HDF5ImageSeriesAdapter(hdf5_file, path='images')

    with pytest.raises(
        IndexError, match='key 1 is out of range for imageseris with length 1'
    ):
        adapter[1]


def test_len_2d(hdf5_file):
    with h5py.File(hdf5_file, 'a') as f:
        del f['images/images']
        f.create_dataset('images/images', data=np.random.rand(10, 10))

    adapter = HDF5ImageSeriesAdapter(hdf5_file, path='images')
    assert len(adapter) == 1


def test_iter_method(hdf5_file):
    adapter = HDF5ImageSeriesAdapter(hdf5_file, path='images')
    iterator = iter(adapter)
    assert isinstance(iterator, ImageSeriesIterator)


def test_setstate_when_attributes_are_present(hdf5_file):
    adapter = HDF5ImageSeriesAdapter(hdf5_file, path='images')
    state = adapter.__getstate__()
    new_adapter = HDF5ImageSeriesAdapter(hdf5_file, path='images')
    new_adapter._meta = {
        'meta1': 'meta1_value'
    }  # set an attribute to simulate existing state
    new_adapter.__setstate__(state)
    assert new_adapter.metadata['meta1'] == 'meta1_value'
    assert new_adapter._images == 'images/images'


def test_iter_method_empty(hdf5_file):
    # Create an empty dataset
    with h5py.File(hdf5_file, 'a') as f:
        del f['images/images']
        f.create_dataset('images/images', data=np.empty((0, 10, 10)))

    adapter = HDF5ImageSeriesAdapter(hdf5_file, path='images')
    assert len(list(iter(adapter))) == 0


def test_shape_property_2d(hdf5_file):
    with h5py.File(hdf5_file, 'a') as f:
        del f['images/images']
        f.create_dataset('images/images', data=np.random.rand(10, 10))

    adapter = HDF5ImageSeriesAdapter(hdf5_file, path='images')
    assert adapter.shape == (10, 10)


def test_getitem_key_zero_2d(hdf5_file):
    with h5py.File(hdf5_file, 'a') as f:
        del f['images/images']
        f.create_dataset('images/images', data=np.random.rand(10, 10))

    adapter = HDF5ImageSeriesAdapter(hdf5_file, path='images')
    assert np.array_equal(adapter[0], np.asarray(adapter._image_dataset))
