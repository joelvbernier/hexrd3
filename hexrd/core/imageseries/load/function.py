"""Adapter class for a custom function that takes an int as an argument
and returns a 2D numpy array.
"""
from . import ImageSeriesAdapter
from ..imageseriesiter import ImageSeriesIterator

import numpy as np


class FunctionImageSeriesAdapter(ImageSeriesAdapter):
    """Collection of Images in numpy array

    Parameters
    ----------
    fname: None
       should be None
    func: a function that returns a numpy array for a index
    num_frames: the number of frames provided by the function
    metadata: dict (optional)
       the metadata dictionary
    """
    format = 'function'

    def __init__(self, fname, **kwargs):
        self._func = kwargs['func']
        self._nframes = kwargs['num_frames']

        self._meta = kwargs.pop('meta', {})

        first_frame = self._first_frame
        self._nxny = first_frame.shape
        self._dtype = first_frame.dtype

    @property
    def metadata(self):
        """Image sequence metadata"""
        return self._meta

    @property
    def _first_frame(self):
        return self._func(0)

    @property
    def shape(self):
        return self._nxny

    @property
    def dtype(self):
        return self._dtype

    def __getitem__(self, key: int) -> np.ndarray:
        if not isinstance(key, int):
            raise TypeError(f"Key must be an integer, but received {type(key).__name__}.")

        if key < 0 or key >= self._nframes:
            raise IndexError(f"Index {key} is out of bounds for axis 0 with size {self._nframes}.")

        return self._func(key)


    def __iter__(self):
        return ImageSeriesIterator(self)

    def __len__(self):
        return self._nframes
