"""Adapter class for numpy array (3D)
"""

from . import ImageSeriesAdapter
from ..imageseriesiter import ImageSeriesIterator

import numpy as np


class ArrayImageSeriesAdapter(ImageSeriesAdapter):
    """ Adapter class for handling a series of images stored in a 3D numpy array.

        This class provides an interface for working with image sequences where 
        the data is organized in a numpy array of shape `(n, m, l)`, where:
        - `n` is the number of images (frames).
        - `m` and `l` are the dimensions of each image.

        Parameters
        ----------
        fname : None
            Placeholder parameter; should be None.
        data : array-like, shape (n, m, l)
            A 3-dimensional numpy array representing the image sequence, where
            the first dimension corresponds to the number of images (frames).
        metadata : dict, optional
            A dictionary containing additional metadata related to the image series. 
            Defaults to an empty dictionary.

        Attributes
        ----------
        format : str
            Specifies the format of the data. In this case, it is set to 'array'.
        
        Raises
        ------
        ValueError
            If the input array does not have 2 or 3 dimensions.
    """

    format = 'array'

    def __init__(self, fname, **kwargs):
        data_arr = np.array(kwargs['data'])
        if data_arr.ndim < 3:
            self._data = np.tile(data_arr, (1, 1, 1))
        elif data_arr.ndim == 3:
            self._data = data_arr
        else:
            raise ValueError(
                'input array must be 2-d or 3-d; you provided ndim=%d'
                % data_arr.ndim
            )

        self._meta = kwargs.pop('meta', dict())
        self._shape = self._data.shape
        self._nframes = self._shape[0]
        self._nxny = self._shape[1:3]

    @property
    def metadata(self):
        """Image sequence metadata"""
        return self._meta

    @property
    def shape(self):
        return self._nxny

    @property
    def dtype(self):
        return self._data.dtype

    def __getitem__(self, key):
        return self._data[key].copy()

    def __iter__(self):
        return ImageSeriesIterator(self)

    def __len__(self):
        return self._nframes
