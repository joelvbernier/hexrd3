import numpy as np
import pytest
from hexrd.core import imageseries
from hexrd.core.imageseries.omega import OmegaSeriesError, OmegaImageSeries


def make_ims(nf, meta):
    """Creates an image series with nf frames and metadata."""
    a = np.zeros((nf, 2, 2))
    ims = imageseries.open(None, 'array', data=a, meta=meta)
    return ims


def test_no_omega():
    """Test that an OmegaSeriesError is raised when there is no omega."""
    ims = make_ims(2, {})
    with pytest.raises(OmegaSeriesError):
        OmegaImageSeries(ims)


def test_nframes_mismatch():
    """Test that an OmegaSeriesError is raised for frame mismatch."""
    metadata = dict(omega=np.zeros((3, 2)))
    ims = make_ims(2, metadata)
    with pytest.raises(OmegaSeriesError):
        OmegaImageSeries(ims)


def test_negative_delta():
    """Test that an OmegaSeriesError is raised when omega delta is negative."""
    omega_data = np.zeros((3, 2))
    omega_data[0, 1] = -0.5
    metadata = dict(omega=omega_data, dtype=float)
    ims = make_ims(3, metadata)
    with pytest.raises(OmegaSeriesError):
        OmegaImageSeries(ims)


def test_one_wedge():
    """Test that a single wedge is detected correctly."""
    nf = 5
    a = np.linspace(0, nf + 1, nf + 1)
    omega_data = np.zeros((nf, 2))
    omega_data[:, 0] = a[:-1]
    omega_data[:, 1] = a[1:]
    metadata = dict(omega=omega_data, dtype=float)
    ims = make_ims(nf, metadata)
    oms = OmegaImageSeries(ims)
    assert oms.nwedges == 1


def test_two_wedges():
    """Test that two wedges are detected correctly."""
    nf = 5
    a = np.linspace(0, nf + 1, nf + 1)
    omega_data = np.zeros((nf, 2))
    omega_data[:, 0] = a[:-1]
    omega_data[:, 1] = a[1:]
    omega_data[3:, :] += 0.1
    metadata = dict(omega=omega_data, dtype=float)
    ims = make_ims(nf, metadata)
    oms = OmegaImageSeries(ims)
    assert oms.nwedges == 2


def test_compare_omegas():
    """Test that omegas from wedges match the original omega data."""
    nf = 5
    a = np.linspace(0, nf + 1, nf + 1)
    omega_data = np.zeros((nf, 2))
    omega_data[:, 0] = a[:-1]
    omega_data[:, 1] = a[1:]
    omega_data[3:, :] += 0.1
    metadata = dict(omega=omega_data, dtype=float)
    ims = make_ims(nf, metadata)
    oms = OmegaImageSeries(ims)

    domega = omega_data - oms.omegawedges.omegas
    dnorm = np.linalg.norm(domega)

    msg = 'omegas from wedges do not match originals'
    assert dnorm == pytest.approx(0.0), msg


def test_wedge_delta():
    """Test that the wedge delta is calculated correctly."""
    nf = 5
    a = np.linspace(0, nf + 1, nf + 1)
    omega_data = np.zeros((nf, 2))
    omega_data[:, 0] = a[:-1]
    omega_data[:, 1] = a[1:]
    omega_data[3:, :] += 0.1
    metadata = dict(omega=omega_data, dtype=float)
    ims = make_ims(nf, metadata)
    oms = OmegaImageSeries(ims)

    my_delta = omega_data[nf - 1, 1] - omega_data[nf - 1, 0]
    wedge = oms.wedge(oms.nwedges - 1)
    assert wedge['delta'] == pytest.approx(my_delta)
