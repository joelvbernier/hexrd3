import abc

from ..imageseriesabc import ImageSeriesABC
from .registry import Registry

# Metaclass for adapter registry


class _RegisterAdapterClass(abc.ABCMeta):
    def __init__(cls, name, bases, attrs):
        abc.ABCMeta.__init__(cls, name, bases, attrs)
        Registry.register(cls)


class ImageSeriesAdapter(ImageSeriesABC, metaclass=_RegisterAdapterClass):

    format = None


# import all adapter modules

from . import (
    array,
    framecache,
    function,
    hdf5,
    imagefiles,
    rawimage,
    metadata,
    trivial,
)

try:
    from dectris.compression import decompress
except ImportError:
    # Dectris compression is not available. Skip the eiger_stream_v1
    pass
else:
    # Eiger stream v1 is supported
    from . import eiger_stream_v1
