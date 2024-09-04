import numpy as np
import pytest

from hexrd import imageseries
from hexrd.core.imageseries import stats
from .common import make_array_ims


def test_stats_average():
    """imageseries.stats: average

    Compares with numpy average
    """
    a, is_a = make_array_ims()
    is_avg = stats.average(is_a)
    np_avg = np.average(a, axis=0).astype(np.float32)
    err = np.linalg.norm(np_avg - is_avg)
    assert err == pytest.approx(0.0), "stats.average failed"
    assert is_avg.dtype == np.float32


def test_stats_median():
    """imageseries.stats: median"""
    a, is_a = make_array_ims()
    ismed = stats.median(is_a)
    amed = np.median(a, axis=0)
    err = np.linalg.norm(amed - ismed)
    assert err == pytest.approx(0.0), "stats.median failed"
    assert ismed.dtype == np.float32


def test_stats_max():
    """imageseries.stats: max"""
    a, is_a = make_array_ims()
    ismax = stats.max(is_a)
    amax = np.max(a, axis=0)
    err = np.linalg.norm(amax - ismax)
    assert err == pytest.approx(0.0), "stats.max failed"
    assert ismax.dtype == is_a.dtype


def test_stats_min():
    """imageseries.stats: min"""
    a, is_a = make_array_ims()
    ismin = stats.min(is_a)
    amin = np.min(a, axis=0)
    err = np.linalg.norm(amin - ismin)
    assert err == pytest.approx(0.0), "stats.min failed"
    assert ismin.dtype == is_a.dtype


def test_stats_percentile():
    """imageseries.stats: percentile"""
    a, is_a = make_array_ims()
    isp90 = stats.percentile(is_a, 90)
    ap90 = np.percentile(a, 90, axis=0).astype(np.float32)
    err = np.linalg.norm(ap90 - isp90)
    assert err == pytest.approx(0.0), "stats.percentile failed"
    assert isp90.dtype == np.float32


# These tests compare chunked operations (iterators) to non-chunked ops

def test_stats_average_chunked():
    """imageseries.stats: chunked average"""
    a, is_a = make_array_ims()
    a_avg = stats.average(a)

    # Run with 1 chunk
    for ismed1 in stats.average_iter(is_a, 1):
        pass
    err = np.linalg.norm(a_avg - ismed1)
    assert err == pytest.approx(0.0), "stats.average failed (1 chunk)"

    # Run with 2 chunks
    for ismed2 in stats.average_iter(is_a, 2):
        pass
    err = np.linalg.norm(a_avg - ismed2)
    assert err == pytest.approx(0.0), "stats.average failed (2 chunks)"


def test_stats_median_chunked():
    """imageseries.stats: chunked median"""
    a, is_a = make_array_ims()
    a_med = stats.median(is_a)

    # Run with 1 chunk
    for ismed1 in stats.median_iter(is_a, 1):
        pass
    err = np.linalg.norm(a_med - ismed1)
    assert err == pytest.approx(0.0), "stats.median failed (1 chunk)"

    # Run with 2 chunks
    for ismed2 in stats.median_iter(is_a, 2):
        pass
    err = np.linalg.norm(a_med - ismed2)
    assert err == pytest.approx(0.0), "stats.median failed (2 chunks)"

    # Run with 3 chunks, with buffer
    for ismed3 in stats.median_iter(is_a, 3, use_buffer=True):
        pass
    err = np.linalg.norm(a_med - ismed3)
    assert err == pytest.approx(0.0), "stats.median failed (3 chunks, buffer)"

    # Run with 3 chunks, no buffer
    for ismed3 in stats.median_iter(is_a, 3, use_buffer=False):
        pass
    err = np.linalg.norm(a_med - ismed3)
    assert err == pytest.approx(0.0), "stats.median failed (3 chunks, no buffer)"
