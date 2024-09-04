import numpy as np
import pytest

from .common import make_array_ims, make_omega_meta, compare

from hexrd.core import imageseries
from hexrd.core.imageseries import process


def run_flip_test(a, flip, aflip):
    """Helper function to run flip tests."""
    is_a = imageseries.open(None, 'array', data=a)
    ops = [('flip', flip)]
    is_p = process.ProcessedImageSeries(is_a, ops)
    is_aflip = imageseries.open(None, 'array', data=aflip)
    diff = compare(is_aflip, is_p)
    assert diff == pytest.approx(0.0), f"flipped [{flip}] image series failed"


def test_process():
    """Processed image series should reproduce the original."""
    _, is_a = make_array_ims()
    is_p = process.ProcessedImageSeries(is_a, [])
    diff = compare(is_a, is_p)
    assert diff == pytest.approx(
        0.0
    ), "processed image series failed to reproduce original"


def test_process_flip_t():
    """Processed image series: flip transpose."""
    flip = 't'
    a, _ = make_array_ims()
    aflip = np.transpose(a, (0, 2, 1))
    run_flip_test(a, flip, aflip)


def test_process_flip_v():
    """Processed image series: flip vertical."""
    flip = 'v'
    a, _ = make_array_ims()
    aflip = a[:, :, ::-1]
    run_flip_test(a, flip, aflip)


def test_process_flip_h():
    """Processed image series: flip horizontal."""
    flip = 'h'
    a, _ = make_array_ims()
    aflip = a[:, ::-1, :]
    run_flip_test(a, flip, aflip)


def test_process_flip_vh():
    """Processed image series: flip vertical + horizontal."""
    flip = 'vh'
    a, _ = make_array_ims()
    aflip = a[:, ::-1, ::-1]
    run_flip_test(a, flip, aflip)


def test_process_flip_r90():
    """Processed image series: flip counterclockwise 90."""
    flip = 'ccw90'
    a, _ = make_array_ims()
    aflip = np.transpose(a, (0, 2, 1))[:, ::-1, :]
    run_flip_test(a, flip, aflip)


def test_process_flip_r270():
    """Processed image series: flip clockwise 90."""
    flip = 'cw90'
    a, _ = make_array_ims()
    aflip = np.transpose(a, (0, 2, 1))[:, :, ::-1]
    run_flip_test(a, flip, aflip)


def test_process_dark():
    """Processed image series: dark image subtraction."""
    a, _ = make_array_ims()
    dark = np.ones_like(a[0])
    is_a = imageseries.open(None, 'array', data=a)
    apos = np.where(a >= 1, a - 1, 0)
    is_a1 = imageseries.open(None, 'array', data=apos)
    ops = [('dark', dark)]
    is_p = process.ProcessedImageSeries(is_a, ops)
    diff = compare(is_a1, is_p)
    assert diff == pytest.approx(0.0), "dark image processing failed"


def test_process_framelist():
    """Processed image series with a subset of frames."""
    a, _ = make_array_ims()
    is_a = imageseries.open(None, 'array', data=a)
    is_a.metadata["omega"] = make_omega_meta(len(is_a))
    frames = [0, 2]
    ops = []
    is_p = process.ProcessedImageSeries(is_a, ops, frame_list=frames)
    is_a2 = imageseries.open(None, 'array', data=a[tuple(frames), ...])
    diff = compare(is_a2, is_p)
    assert diff == pytest.approx(0.0), "frame list processing failed"
    assert len(is_p) == len(is_p.metadata["omega"])


def test_process_shape():
    """Test that the processed image series has the correct shape."""
    a, _ = make_array_ims()
    is_a = imageseries.open(None, 'array', data=a)
    ops = []
    is_p = process.ProcessedImageSeries(is_a, ops)
    pshape = is_p.shape
    fshape = is_p[0].shape
    for i in range(2):
        assert fshape[i] == pshape[i], f"Shape mismatch at index {i}"


def test_process_dtype():
    """Test that the processed image series retains the correct dtype."""
    a, _ = make_array_ims()
    is_a = imageseries.open(None, 'array', data=a)
    ops = []
    is_p = process.ProcessedImageSeries(is_a, ops)
    assert (
        is_p.dtype == is_p[0].dtype
    ), "dtype mismatch between processed image and original"
