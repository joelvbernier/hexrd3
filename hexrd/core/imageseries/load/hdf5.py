"""HDF5 adapter class
"""

import h5py
import warnings

import numpy as np

from . import ImageSeriesAdapter
from ..imageseriesiter import ImageSeriesIterator


class HDF5ImageSeriesAdapter(ImageSeriesAdapter):
    """Collection of Images in HDF5 Format

    Parameters
    ----------
    fname : str or h5py.File object
        filename of the HDF5 file, or an open h5py file.  Note that this
        class will close the h5py.File when finished.
    path : str, required
        The path to the HDF dataset containing the image data
    dataname : str, optional
        The name of the HDF dataset containing the 2-d or 3d image data.
        The default values is 'images'.
    """

    format = 'hdf5'

    def __init__(self, fname, **kwargs):
        if isinstance(fname, h5py.File):
            self._h5name = fname.filename
            self._h5file = fname
        else:
            self._h5name = fname
            self._h5file = h5py.File(self._h5name, 'r')

        self._path = kwargs['path']
        self._dataname = kwargs.pop('dataname', 'images')
        self._images = '/'.join([self._path, self._dataname])
        self._load_data()
        self._meta = self._getmeta()

    def close(self):
        self._image_dataset = None
        self._data_group = None
        self._h5file.close()
        self._h5file = None

    def __del__(self):
        # !!! Note this is not ideal, as the use of __del__ is problematic.
        #     However, it is highly unlikely that the usage of a ImageSeries
        #     would pose a problem.  A warning will (hopefully) be emitted if
        #     an issue arises at some point
        try:
            self.close()
        except Exception:
            warnings.warn("HDF5ImageSeries could not close h5 file")

    def __getitem__(self, key):
        if self._ndim == 2:
            if key != 0:
                raise IndexError(
                    f'key {key} is out of range for imageseris with length 1'
                )
            # !!! necessary when not returning a slice
            return np.asarray(self._image_dataset)
        else:
            return self._image_dataset[key]

    def __iter__(self):
        return ImageSeriesIterator(self)

    def __len__(self):
        if self._ndim == 2:
            return 1
        else:
            # !!! must be 3-d; exception handled in load_data()
            return len(self._image_dataset)

    def __getstate__(self):
        # Remove any non-pickleable attributes
        to_remove = [
            '_h5file',
            '_image_dataset',
            '_data_group',
        ]

        # Make a copy of the dict to modify
        state = self.__dict__.copy()

        # Remove them
        for attr in to_remove:
            state.pop(attr, None)

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._h5file = h5py.File(self._h5name, 'r')
        self._load_data()

    def _load_data(self):
        self._image_dataset = self._h5file[self._images]
        self._ndim = self._image_dataset.ndim
        if self._ndim not in [2, 3]:
            raise RuntimeError(
                f'Image data must be a 2-d or 3-d array; yours is {self._ndim}'
            )
        self._data_group = self._h5file[self._path]

    def _getmeta(self):
        mdict = {}
        for k, v in list(self._data_group.attrs.items()):
            mdict[k] = v

        return mdict

    @property
    def metadata(self):
        """(read-only) Image sequence metadata

        note: metadata loaded on open and allowed to be modified
        """
        return self._meta

    @property
    def dtype(self):
        return self._image_dataset.dtype

    @property
    def shape(self):
        if self._ndim == 2:
            return self._image_dataset.shape
        else:
            return self._image_dataset.shape[1:]
