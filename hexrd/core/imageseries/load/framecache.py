import os
import numpy as np
from scipy.sparse import csr_matrix  # type: ignore
import yaml
from typing import Dict, Set

from . import ImageSeriesAdapter
from ..imageseriesiter import ImageSeriesIterator
from .metadata import yamlmeta


class FrameCacheImageSeriesAdapter(ImageSeriesAdapter):
    """Adapter for handling image series stored in frame cache format."""

    format = 'frame-cache'

    def __init__(self, filename: str, style: str = 'npz', **kwargs):
        """Initialize the FrameCacheImageSeriesAdapter.

        Args:
            filename (str): The path to the frame cache or YAML file.
            style (str): The format style, either 'npz' or 'yaml'.
        """
        self._filename: str = filename
        self._meta: Dict[str, any] = {}

        if style.lower() in ('yml', 'yaml', 'test'):
            self._load_metadata_from_yaml()
            self._load_cache(from_yaml=True)
        else:
            self._load_cache()

    def _load_metadata_from_yaml(self) -> None:
        """Load metadata from a YAML file."""
        with open(self._filename, "r") as file:
            data = yaml.safe_load(file)
        data_info = data['data']
        self._cache_file: str = data_info['file']
        self._nframes: int = data_info['nframes']
        self._shape: tuple = tuple(data_info['shape'])
        self._dtype: np.dtype = np.dtype(data_info['dtype'])
        self._meta: Dict[str, any] = yamlmeta(data['meta'], path=self._cache_file)

    def _load_cache(self, from_yaml: bool = False) -> None:
        """Load frame data into a list of CSR sparse matrices.

        Args:
            from_yaml (bool): Whether the cache is being loaded from a YAML file.
        """
        self._frames: list = []
        cache_path: str = self._get_cache_path(from_yaml)

        array_data = np.load(cache_path)
        self._nframes: int = int(array_data['nframes'])
        self._shape: tuple = tuple(array_data['shape'])
        self._dtype: np.dtype = (
            np.dtype(array_data['dtype'].item())
            if isinstance(array_data['dtype'], np.ndarray)
            else np.dtype(array_data['dtype'])
        )

        for i in range(self._nframes):
            row = array_data[f"{i}_row"]
            col = array_data[f"{i}_col"]
            data = array_data[f"{i}_data"]
            frame = csr_matrix(
                (data, (row, col)), shape=self._shape, dtype=self._dtype
            )
            self._frames.append(frame)

        if not from_yaml:
            self._meta = {
                key: array_data[key]
                for key in array_data
                if key not in self._get_frame_keys()
            }

    def _get_cache_path(self, from_yaml: bool) -> str:
        """Get the full path to the cache file.

        Args:
            from_yaml (bool): Whether the path is derived from a YAML file.

        Returns:
            str: The full path to the cache file.
        """
        if from_yaml:
            base_path: str = os.path.dirname(self._filename)
            return (
                os.path.join(base_path, self._cache_file)
                if not os.path.isabs(self._cache_file)
                else self._cache_file
            )
        return self._filename

    def _get_frame_keys(self) -> Set[str]:
        """Generate keys used for frames in the NPZ file.

        Returns:
            set: A set of keys related to frame data.
        """
        keys: Set[str] = set()
        for i in range(self._nframes):
            keys.update({f"{i}_row", f"{i}_col", f"{i}_data"})
        return keys

    @property
    def metadata(self) -> Dict[str, any]:
        """dict: Image sequence metadata."""
        return self._meta

    @property
    def dtype(self) -> np.dtype:
        """numpy.dtype: Data type of the frames."""
        return self._dtype

    @property
    def shape(self) -> tuple:
        """tuple: Shape of the frames."""
        return self._shape

    def __getitem__(self, index: int) -> np.ndarray:
        """Retrieve a frame by its index.

        Args:
            index (int): Index of the frame to retrieve.

        Returns:
            numpy.ndarray: The requested frame as a dense array.
        """
        return self._frames[index].toarray()

    def __iter__(self) -> ImageSeriesIterator:
        """Return an iterator over the frames.

        Returns:
            ImageSeriesIterator: Iterator over the image series.
        """
        return ImageSeriesIterator(self)

    def __len__(self) -> int:
        """Return the number of frames in the series.

        Returns:
            int: Number of frames.
        """
        return self._nframes
