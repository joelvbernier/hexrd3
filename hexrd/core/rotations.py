"""
-*- coding: utf-8 -*-
=============================================================================
Copyright (c) 2012, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.
Written by Joel Bernier <bernier2@llnl.gov> and others.
LLNL-CODE-529294.
All rights reserved.

This file is part of HEXRD. For details on dowloading the source,
see the file COPYING.

Please also see the file LICENSE.

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License (as published by the Free
Software Foundation) version 2.1 dated February 1999.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the
GNU General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this program (see file LICENSE); if not, write to
the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
Boston, MA 02111-1307 USA or visit <http://www.gnu.org/licenses/>.
=============================================================================

Module containing functions relevant to rotations
"""
import sys

import numpy as np
from scipy.optimize import leastsq
from scipy.spatial.transform import Rotation as R
from hexrd.core import constants as cnst
from hexrd.core.matrixutil import (
    unit_vector,
    mult_mat_array,
    null_space,
    find_duplicate_vectors,
)

# =============================================================================
# Module Data
# =============================================================================

period_dict = cnst.PERIOD_DICT
conversion_to_dict = {'degrees': 180 / np.pi, 'radians': np.pi / 180}

I3 = np.eye(3)  # (3, 3) identity matrix

# axes orders, all permutations
axes_orders = [
    'xyz',
    'zyx',
    'zxy',
    'yxz',
    'yzx',
    'xzy',
    'xyx',
    'xzx',
    'yxy',
    'yzy',
    'zxz',
    'zyz',
]

sq3by2 = np.sqrt(3.0) / 2.0
piby2 = np.pi / 2.0
piby3 = np.pi / 3.0
piby4 = np.pi / 4.0
piby6 = np.pi / 6.0

# =============================================================================
# Functions
# =============================================================================


def arccos_safe(cosines):
    """
    Protect against numbers slightly larger than 1 in magnitude
    due to round-off
    """
    cosines = np.atleast_1d(cosines)
    if (np.abs(cosines) > 1.00001).any():
        print(f"attempt to take arccos of {cosines}", file=sys.stderr)
        raise RuntimeError("unrecoverable error")
    return np.arccos(np.clip(cosines, -1.0, 1.0))


#
#  ==================== Quaternions
#


def _quat_to_scipy_rotation(q: np.ndarray) -> R:
    """
    Scipy has quaternions in a differnt order, this method converts them
    q must be a 2d array of shape (4, n).
    """
    return R.from_quat(np.roll(q.T, -1, axis=1))


def _scipy_rotation_to_quat(r: R) -> np.ndarray:
    quat = np.roll(np.atleast_2d(r.as_quat()), 1, axis=1).T
    # Fix quat would work, but it does too much.  Only need to check positive
    quat *= np.sign(quat[0, :])
    return quat


def fix_quat(q):
    """
    flip to positive q0 and normalize
    """
    qdims = q.ndim
    if qdims == 3:
        l, m, n = q.shape
        assert m == 4, 'your 3-d quaternion array isn\'t the right shape'
        q = q.transpose(0, 2, 1).reshape(l * n, 4).T

    qfix = unit_vector(q)

    q0negative = qfix[0,] < 0
    qfix[:, q0negative] = -1 * qfix[:, q0negative]

    if qdims == 3:
        qfix = qfix.T.reshape(l, n, 4).transpose(0, 2, 1)

    return qfix


def invert_quat(q):
    """
    silly little routine for inverting a quaternion
    """
    numq = q.shape[1]

    imat = np.tile(np.vstack([-1, 1, 1, 1]), (1, numq))

    qinv = imat * q

    return fix_quat(qinv)


def misorientation(q1, q2, symmetries=None):
    """
    PARAMETERS
    ----------
    q1: array(4, 1)
        a single quaternion
    q2: array(4, n)
        array of quaternions
    symmetries: tuple, optional
        1- or 2-tuple with symmetries (quaternion arrays);
        for crystal symmetry only, use a 1-tuple;
        with both crystal and sample symmetry use a 2-tuple
        Default is no symmetries.

    RETURNS
    -------
    angle: array(n)
        the misorientation angle between `q1` and each quaternion in `q2`
    mis: array(4, n)
        the quaternion of the smallest misorientation angle
    """
    if not isinstance(q1, np.ndarray) or not isinstance(q2, np.ndarray):
        raise RuntimeError("quaternion args are not of type `numpy ndarray'")

    if q1.ndim != 2 or q2.ndim != 2:
        raise RuntimeError(
            "quaternion args are the wrong shape; must be 2-d (columns)"
        )

    if q1.shape[1] != 1:
        raise RuntimeError(
            f"first argument should be a single quaternion, got shape {q1.shape}"
        )

    if symmetries is None:
        # no symmetries; use identity
        symmetries = (np.c_[1.0, 0, 0, 0].T, np.c_[1.0, 0, 0, 0].T)
    else:
        # check symmetry argument
        if len(symmetries) == 1:
            if not isinstance(symmetries[0], np.ndarray):
                raise RuntimeError("symmetry argument is not an numpy array")
            else:
                # add triclinic sample symmetry (identity)
                symmetries += (np.c_[1.0, 0, 0, 0].T,)
        elif len(symmetries) == 2:
            if not isinstance(symmetries[0], np.ndarray) or not isinstance(
                symmetries[1], np.ndarray
            ):
                raise RuntimeError(
                    "symmetry arguments are not an numpy arrays"
                )
        elif len(symmetries) > 2:
            raise RuntimeError(
                f"symmetry argument has {len(symmetries)} entries; should be 1 or 2"
            )

    # set some lengths
    n = q2.shape[1]  # length of misorientation list
    m = symmetries[0].shape[1]  # crystal (right)
    p = symmetries[1].shape[1]  # sample  (left)

    # tile q1 inverse
    q1i = quat_product_matrix(invert_quat(q1), mult='right').squeeze()

    # convert symmetries to (4, 4) qprod matrices
    rsym = quat_product_matrix(symmetries[0], mult='right')
    lsym = quat_product_matrix(symmetries[1], mult='left')

    # Do R * Gc, store as
    # [q2[:, 0] * Gc[:, 0:m], ..., q2[:, n-1] * Gc[:, 0:m]]
    q2 = np.dot(rsym, q2).transpose(2, 0, 1).reshape(m * n, 4).T

    # Do Gs * (R * Gc), store as
    # [Gs[:, 0:p]*q[:,   0]*Gc[:, 0], ..., Gs[:, 0:p]*q[:,   0]*Gc[:, m-1], ...
    #  Gs[:, 0:p]*q[:, n-1]*Gc[:, 0], ..., Gs[:, 0:p]*q[:, n-1]*Gc[:, m-1]]
    q2 = np.dot(lsym, q2).transpose(2, 0, 1).reshape(p * m * n, 4).T

    # Calculate the class misorientations for full symmetrically equivalent
    # classes for q1 and q2.  Note the use of the fact that the application
    # of the symmetry groups is an isometry.
    eqv_mis = fix_quat(np.dot(q1i, q2))

    # Reshape scalar comp columnwise by point in q2 (and q1, if applicable)
    scl_eqv_mis = eqv_mis[0, :].reshape(n, p * m).T

    # Find misorientation closest to origin for each n equivalence classes
    #   - fixed quats so garaunteed that sclEqvMis is nonnegative
    qmax = scl_eqv_mis.max(0)

    # remap indices to use in eqvMis
    qmax_ind = (scl_eqv_mis == qmax).nonzero()
    qmax_ind = np.c_[qmax_ind[0], qmax_ind[1]]

    eqv_mis_col_ind = np.sort(qmax_ind[:, 0] + qmax_ind[:, 1] * p * m)

    # store Rmin in q
    mis = eqv_mis[np.ix_(list(range(4)), eqv_mis_col_ind)]

    angle = 2 * arccos_safe(qmax)

    return angle, mis


def quat_product(q1, q2):
    """
    Product of two unit quaternions.

    qp = quat_product(q2, q1)

    q2, q1 are 4 x n, arrays whose columns are
           quaternion parameters

    qp is 4 x n, an array whose columns are the
       quaternion parameters of the product; the
       first component of qp is nonnegative

    If R(q) is the rotation corresponding to the
    quaternion parameters q, then

    R(qp) = R(q2) R(q1)
    """
    rot_1 = _quat_to_scipy_rotation(q1)
    rot_2 = _quat_to_scipy_rotation(q2)
    rot_p = rot_2 * rot_1
    return _scipy_rotation_to_quat(rot_p)


def quat_product_matrix(quats, mult='right'):
    """
    Form 4 x 4 arrays to perform the quaternion product

    USAGE
        qmats = quat_product_matrix(quats, mult='right')

    INPUTS
        1) quats is (4, n), a numpy ndarray array of n quaternions
           horizontally concatenated
        2) mult is a keyword arg, either 'left' or 'right', denoting
           the sense of the multiplication:

                       | quat_product_matrix(h, mult='right') * q
           q * h  --> <
                       | quat_product_matrix(q, mult='left') * h

    OUTPUTS
        1) qmats is (n, 4, 4), the left or right quaternion product
           operator

    NOTES
       *) This function is intended to replace a cross-product based
          routine for products of quaternions with large arrays of
          quaternions (e.g. applying symmetries to a large set of
          orientations).
    """

    if quats.shape[0] != 4:
        raise RuntimeError("input is the wrong size along the 0-axis")

    nq = quats.shape[1]
    q0 = quats[0, :].copy()
    q1 = quats[1, :].copy()
    q2 = quats[2, :].copy()
    q3 = quats[3, :].copy()
    if mult == 'right':
        qmats = np.array(
            [
                [q0],
                [q1],
                [q2],
                [q3],
                [-q1],
                [q0],
                [-q3],
                [q2],
                [-q2],
                [q3],
                [q0],
                [-q1],
                [-q3],
                [-q2],
                [q1],
                [q0],
            ]
        )
    elif mult == 'left':
        qmats = np.array(
            [
                [q0],
                [q1],
                [q2],
                [q3],
                [-q1],
                [q0],
                [q3],
                [-q2],
                [-q2],
                [-q3],
                [q0],
                [q1],
                [-q3],
                [q2],
                [-q1],
                [q0],
            ]
        )
    # some fancy reshuffling...
    qmats = qmats.T.reshape((nq, 4, 4)).transpose(0, 2, 1)
    return qmats


def angle_axis_to_quat(angle, rotaxis):
    """
    make an hstacked array of quaternions from arrays of angle/axis pairs
    """
    angle = np.atleast_1d(angle)
    n = len(angle)

    if rotaxis.shape[1] == 1:
        rotaxis = np.tile(rotaxis, (1, n))
    elif rotaxis.shape[1] != n:
        raise RuntimeError("rotation axes argument has incompatible shape")

    # Normalize the axes
    rotaxis = unit_vector(rotaxis)
    rot = R.from_rotvec((angle * rotaxis).T)
    return _scipy_rotation_to_quat(rot)


def exp_map_to_quat(exp_maps):
    """
    Returns the unit quaternions associated with exponential map parameters.

    Parameters
    ----------
    exp_maps : array_like
        The (3,) or (3, n) list of hstacked exponential map parameters to
        convert.

    Returns
    -------
    quats : array_like
        The (4,) or (4, n) array of unit quaternions.

    Notes
    -----
    1) be aware that the output will always have non-negative q0; recall the
       antipodal symmetry of unit quaternions

    """
    cdim = 3  # critical dimension of input
    exp_maps = np.atleast_2d(exp_maps)
    if len(exp_maps) == 1:
        assert (
            exp_maps.shape[1] == cdim
        ), f"your input quaternion must have {cdim} elements"
        exp_maps = np.reshape(exp_maps, (cdim, 1))
    else:
        assert (
            len(exp_maps) == cdim
        ), f"your input quaternions must have shape ({cdim}, n) for n > 1"

    return _scipy_rotation_to_quat(R.from_rotvec(exp_maps.T)).squeeze()


def rot_mat_to_quat(r_mat):
    """
    Generate quaternions from rotation matrices
    """
    return _scipy_rotation_to_quat(R.from_matrix(r_mat))


def quat_average_cluster(q_in, qsym):
    """
    Average two quaternions with symmetries
    """
    assert q_in.ndim == 2, 'input must be 2-s hstacked quats'

    # renormalize
    q_in = unit_vector(q_in)

    # check to see num of quats is > 1
    if q_in.shape[1] < 3:
        if q_in.shape[1] == 1:
            q_bar = q_in
        else:
            ma, mq = misorientation(
                q_in[:, 0].reshape(4, 1), q_in[:, 1].reshape(4, 1), (qsym,)
            )

            q_bar = quat_product(
                q_in[:, 0].reshape(4, 1),
                exp_map_to_quat(0.5 * ma * unit_vector(mq[1:])).reshape(4, 1),
            )
    else:
        # first drag to origin using first quat (arb!)
        q0 = q_in[:, 0].reshape(4, 1)
        qrot = np.dot(quat_product_matrix(invert_quat(q0), mult='left'), q_in)

        # second, re-cast to FR
        qrot = to_fundamental_region(qrot.squeeze(), crys_sym=qsym)

        # compute arithmetic average
        q_bar = unit_vector(np.average(qrot, axis=1).reshape(4, 1))

        # unrotate!
        q_bar = np.dot(quat_product_matrix(q0, mult='left'), q_bar)

        # re-map
        q_bar = to_fundamental_region(q_bar, crys_sym=qsym)
    return q_bar


def quat_average(q_in, qsym):
    """
    Average two quaternions with symmetries
    """
    assert q_in.ndim == 2, 'input must be 2-s hstacked quats'

    # renormalize
    q_in = unit_vector(q_in)

    # check to see num of quats is > 1
    if q_in.shape[1] < 3:
        if q_in.shape[1] == 1:
            q_bar = q_in
        else:
            ma, mq = misorientation(
                q_in[:, 0].reshape(4, 1), q_in[:, 1].reshape(4, 1), (qsym,)
            )
            q_bar = quat_product(
                q_in[:, 0].reshape(4, 1),
                exp_map_to_quat(0.5 * ma * unit_vector(mq[1:].reshape(3, 1))),
            )
    else:
        # use first quat as initial guess
        phi = 2.0 * np.arccos(q_in[0, 0])
        if phi <= np.finfo(float).eps:
            x0 = np.zeros(3)
        else:
            n = unit_vector(q_in[1:, 0].reshape(3, 1))
            x0 = phi * n.flatten()

        # Objective function to optimize
        def _quat_average_obj(xi_in, quats, qsym):
            phi = np.sqrt(sum(xi_in.flatten() * xi_in.flatten()))
            if phi <= np.finfo(float).eps:
                q0 = np.c_[1.0, 0.0, 0.0, 0.0].T
            else:
                n = xi_in.flatten() / phi
                q0 = np.hstack([np.cos(0.5 * phi), np.sin(0.5 * phi) * n])
            resd = misorientation(q0.reshape(4, 1), quats, (qsym,))[0]
            return resd

        # Optimize
        results = leastsq(_quat_average_obj, x0, args=(q_in, qsym))
        phi = np.sqrt(sum(results[0] * results[0]))
        if phi <= np.finfo(float).eps:
            q_bar = np.c_[1.0, 0.0, 0.0, 0.0].T
        else:
            n = results[0] / phi
            q_bar = np.hstack(
                [np.cos(0.5 * phi), np.sin(0.5 * phi) * n]
            ).reshape(4, 1)
    return q_bar


def quat_to_exp_map(quats):
    """
    Return the exponential map parameters for an array of unit quaternions

    Parameters
    ----------
    quats : array_like
        The (4, ) or (4, n) array of hstacked unit quaternions.  The convention
        is [q0, q] where q0 is the scalar part and q is the vector part.

    Returns
    -------
    expmaps : array_like
        The (3, ) or (3, n) array of exponential map parameters associated
        with the input quaternions.

    """
    cdim = 4  # critical dimension of input
    quats = np.atleast_2d(quats)
    if len(quats) == 1:
        assert (
            quats.shape[1] == cdim
        ), f"your input quaternion must have {cdim} elements"
        quats = np.reshape(quats, (cdim, 1))
    else:
        assert (
            len(quats) == cdim
        ), f"your input quaternions must have shape ({cdim}, n) for n > 1"

    return _quat_to_scipy_rotation(quats).as_rotvec().T.squeeze()


def exp_map_to_rot_mat(exp_map):
    """
    Make a rotation matrix from an expmap
    """
    if exp_map.ndim == 1:
        exp_map = exp_map.reshape(3, 1)

    return R.from_rotvec(exp_map.T).as_matrix().squeeze()


def quat_to_rot_mat(quat):
    """
    Convert quaternions to rotation matrices.

    Take an array of n quats (numpy ndarray, 4 x n) and generate an
    array of rotation matrices (n x 3 x 3)

    Parameters
    ----------
    quat : TYPE
        DESCRIPTION.

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    rmat : TYPE
        DESCRIPTION.

    Notes
    -----
    Uses the truncated series expansion for the exponential map;
    didvide-by-zero is checked using the global 'cnst.epsf'
    """
    if quat.ndim == 1:
        if len(quat) != 4:
            raise RuntimeError("input is the wrong shape")
        quat = quat.reshape(4, 1)
    elif quat.shape[0] != 4:
        raise RuntimeError("input is the wrong shape")

    rmat = _quat_to_scipy_rotation(quat).as_matrix()

    return np.squeeze(rmat)


def rot_mat_to_angle_axis(rot_mat):
    """
    Extracts angle and axis invariants from rotation matrices.

    Parameters
    ----------
    R : numpy.ndarray
        The (3, 3) or (n, 3, 3) array of rotation matrices.
        Note that these are assumed to be proper orthogonal.

    Raises
    ------
    RuntimeError
        If `R` is not an shape is not (3, 3) or (n, 3, 3).

    Returns
    -------
    phi : numpy.ndarray
        The (n, ) array of rotation angles for the n input
        rotation matrices.
    n : numpy.ndarray
        The (3, n) array of unit rotation axes for the n
        input rotation matrices.

    """
    if not isinstance(rot_mat, np.ndarray):
        raise RuntimeError('Input must be a 2 or 3-d ndarray')
    else:
        rdim = rot_mat.ndim
        if rdim == 2:
            rot_mat = np.tile(rot_mat, (1, 1, 1))
        elif rdim == 3:
            pass
        else:
            raise RuntimeError(
                "rot_mat array must be (3, 3) or (n, 3, 3); "
                f"input has dimension {rdim}"
            )

    rot_vec = R.from_matrix(rot_mat).as_rotvec()
    angs = np.linalg.norm(rot_vec, axis=1)
    axes = unit_vector(rot_vec.T)
    return angs, axes


def _check_axes_order(x):
    if not isinstance(x, str):
        raise RuntimeError("argument must be str")
    axo = x.lower()
    if axo not in axes_orders:
        raise RuntimeError(f"order '{x}' is not a valid choice")
    return axo


def _check_is_rmat(x):
    x = np.asarray(x)
    if x.shape != (3, 3):
        raise RuntimeError("shape of input must be (3, 3)")
    chk1 = np.linalg.det(x)
    chk2 = np.sum(np.abs(np.eye(3) - np.dot(x, x.T)))
    if 1.0 - np.abs(chk1) < cnst.SQRT_EPSF and chk2 < cnst.SQRT_EPSF:
        return x
    raise RuntimeError("input is not an orthogonal matrix")


def make_rmat_euler(tilt_angles, axes_order, extrinsic=True):
    """
    Generate rotation matrix from Euler angles.

    Parameters
    ----------
    tilt_angles : array_like
        The (3, ) list of Euler angles in RADIANS.
    axes_order : str
        The axes order specification (case-insensitive).  This must be one
        of the following: 'xyz', 'zyx'
                          'zxy', 'yxz'
                          'yzx', 'xzy'
                          'xyx', 'xzx'
                          'yxy', 'yzy'
                          'zxz', 'zyz'
    extrinsic : bool, optional
        Flag denoting the convention.  If True, the convention is
        extrinsic (passive); if False, the convention is
        instrinsic (active). The default is True.

    Returns
    -------
    numpy.ndarray
        The (3, 3) rotation matrix corresponding to the input specification.

    TODO: add kwarg for unit selection for `tilt_angles`
    TODO: input checks
    """
    axo = _check_axes_order(axes_order)
    if not extrinsic:
        axo = axo.upper()

    return R.from_euler(axo, tilt_angles).as_matrix()


class RotMatEuler:
    """
    Conversion object for euler angles and rotation matrices
    """

    def __init__(
        self, angles, axes_order, extrinsic=True, units=cnst.ANGULAR_UNITS
    ):
        """
        Abstraction of a rotation matrix defined by Euler angles.

        Parameters
        ----------
        angles : array_like
            The (3, ) list of Euler angles in RADIANS.
        axes_order : str
            The axes order specification (case-insensitive).  This must be one
            of the following:

                'xyz', 'zyx'
                'zxy', 'yxz'
                'yzx', 'xzy'
                'xyx', 'xzx'
                'yxy', 'yzy'
                'zxz', 'zyz'

        extrinsic : bool, optional
            Flag denoting the convention.  If True, the convention is
            extrinsic (passive); if False, the convention is
            instrinsic (active). The default is True.

        Returns
        -------
        None.

        TODO: add check that angle input is array-like, len() = 3?
        TODO: add check on extrinsic as bool
        """
        self._axes = np.eye(3)
        self._axes_dict = dict(x=0, y=1, z=2)

        # these will be properties
        self._angles = angles
        self._axes_order = _check_axes_order(axes_order)
        self._extrinsic = extrinsic
        if units.lower() not in period_dict:
            raise RuntimeError(f"angular units '{units}' not understood")
        self._units = units

    @property
    def angles(self):
        """
        Euler angles, in whatever units are set in units, defaults to radians
        """
        return self._angles

    @angles.setter
    def angles(self, x):
        x = np.atleast_1d(x).flatten()
        if len(x) == 3:
            self._angles = x
        else:
            raise RuntimeError("input must be array-like with __len__ = 3")

    @property
    def axes_order(self):
        """
        Euler axes order, like 'xyx', 'zyx', etc.
        """
        return self._axes_order

    @axes_order.setter
    def axes_order(self, x):
        axo = _check_axes_order(x)
        self._axes_order = axo

    @property
    def extrinsic(self):
        """
        True if euler andles are extrinsic, False if intrinsic
        """
        return self._extrinsic

    @extrinsic.setter
    def extrinsic(self, x):
        if isinstance(x, bool):
            self._extrinsic = x
        else:
            raise RuntimeError("input must be a bool")

    @property
    def units(self):
        """
        Radians or degrees, what the euler angles are
        """
        return self._units

    @units.setter
    def units(self, x):
        if isinstance(x, str) and x in period_dict:
            if self._units != x:
                # !!! we are changing units; update self.angles
                self.angles = conversion_to_dict[x] * np.asarray(self.angles)
            self._units = x
        else:
            raise RuntimeError("input must be 'degrees' or 'radians'")

    @property
    def rmat(self):
        """
        Return the rotation matrix.

        As calculated from angles, axes_order, and convention.

        Returns
        -------
        numpy.ndarray
            The (3, 3) proper orthogonal matrix according to the specification.

        """
        angs_in = self.angles
        if self.units == 'degrees':
            angs_in = conversion_to_dict['radians'] * angs_in
        self._rmat = make_rmat_euler(angs_in, self.axes_order, self.extrinsic)
        return self._rmat

    @rmat.setter
    def rmat(self, x):
        """
        Update class via input rotation matrix.

        Parameters
        ----------
        x : array_like
            A (3, 3) array to be interpreted as a rotation matrix.

        Returns
        -------
        None
        """
        rmat = _check_is_rmat(x)
        self._rmat = rmat

        axo = self.axes_order
        if not self.extrinsic:
            axo = axo.upper()
        self._angles = R.from_matrix(rmat).as_euler(
            axo, self.units == 'degrees'
        )

    @property
    def exponential_map(self):
        """
        The matrix invariants of self.rmat as exponential map parameters

        Returns
        -------
        np.ndarray
            The (3, ) array representing the exponential map parameters of
            the encoded rotation (self.rmat).

        """
        phi, n = rot_mat_to_angle_axis(self.rmat)
        return phi * n.flatten()

    @exponential_map.setter
    def exponential_map(self, x):
        """
        Updates encoded rotation via exponential map parameters

        Parameters
        ----------
        x : array_like
            The (3, ) vector representing exponential map parameters of a
            rotation.

        Returns
        -------
        None.

        Notes
        -----
        Updates the encoded rotation from expoential map parameters via
        self.rmat property
        """
        x = np.atleast_1d(x).flatten()
        assert len(x) == 3, "input must have exactly 3 elements"
        self.rmat = exp_map_to_rot_mat(x.reshape(3, 1))  # use local func

#
#  ==================== Utility Functions
#


def map_angle(ang, ang_range=None, units=cnst.ANGULAR_UNITS):
    """
    Utility routine to map an angle into a specified period
    """
    if units.lower() == 'degrees':
        period = 360.0
    elif units.lower() == 'radians':
        period = 2.0 * np.pi
    else:
        raise RuntimeError("unknown angular units: " + units)

    ang = np.nan_to_num(np.atleast_1d(np.float_(ang)))

    min_val = -period / 2
    max_val = period / 2

    # if we have a specified angular range, use that
    if ang_range is not None:
        ang_range = np.atleast_1d(np.float_(ang_range))

        min_val = ang_range.min()
        max_val = ang_range.max()

        if not np.allclose(max_val - min_val, period):
            raise RuntimeError('range is incomplete!')

    val = np.mod(ang - min_val, max_val - min_val) + min_val
    # To match old implementation, map to closer value on the boundary
    # Not doing this breaks hedm_instrument's _extract_polar_maps
    val[np.logical_and(val == min_val, ang > min_val)] = max_val
    return val


def angular_difference(ang_list0, ang_list1, units=cnst.ANGULAR_UNITS):
    """
    Do the proper (acute) angular difference in the context of a branch cut.

    *) Default angular range in the code is [-pi, pi]
    """
    period = period_dict[units]
    d = np.abs(ang_list1 - ang_list0)
    return np.minimum(d, period - d)


def apply_sym(vec, qsym, cs_flag=False, ignore_sign=False, tol=cnst.SQRT_EPSF):
    """
    Apply symmetry group to a single 3-vector (columnar) argument.

    csFlag : centrosymmetry flag
    cullPM : cull +/- flag
    """
    nsym = qsym.shape[1]
    r_sym = quat_to_rot_mat(qsym)
    if nsym == 1:
        r_sym = np.array(
            [
                r_sym,
            ]
        )
    allhkl = (
        mult_mat_array(r_sym, np.tile(vec, (nsym, 1, 1)))
        .swapaxes(1, 2)
        .reshape(nsym, 3)
        .T
    )

    if cs_flag:
        allhkl = np.hstack([allhkl, -1 * allhkl])
    _, uid = find_duplicate_vectors(allhkl, tol=tol, ignore_sign=ignore_sign)

    return allhkl[np.ix_(list(range(3)), uid)]


# =============================================================================
# Symmetry functions
# =============================================================================


def to_fundamental_region(q, crys_sym='Oh', samp_sym=None):
    """
    Map quaternions to fundamental region.

    Parameters
    ----------
    q : TYPE
        DESCRIPTION.
    crys_sym : TYPE, optional
        DESCRIPTION. The default is 'Oh'.
    samp_sym : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    qr : TYPE
        DESCRIPTION.
    """
    qdims = q.ndim
    if qdims == 3:
        l3, m3, n3 = q.shape
        assert m3 == 4, 'your 3-d quaternion array isn\'t the right shape'
        q = q.transpose(0, 2, 1).reshape(l3 * n3, 4).T
    if isinstance(crys_sym, str):
        qsym_c = quat_product_matrix(
            laue_group_to_quat(crys_sym), 'right'
        )  # crystal symmetry operator
    else:
        qsym_c = quat_product_matrix(crys_sym, 'right')

    n = q.shape[1]  # total number of quats
    m = qsym_c.shape[0]  # number of symmetry operations

    #
    # MAKE EQUIVALENCE CLASS
    #
    # Do R * Gc, store as
    # [q[:, 0] * Gc[:, 0:m], ..., 2[:, n-1] * Gc[:, 0:m]]
    qeqv = np.dot(qsym_c, q).transpose(2, 0, 1).reshape(m * n, 4).T

    if samp_sym is None:
        # need to fix quats to sort
        qeqv = fix_quat(qeqv)

        # Reshape scalar comp columnwise by point in qeqv
        q0 = qeqv[0, :].reshape(n, m).T

        # Find q0 closest to origin for each n equivalence classes
        q0max_col_ind = np.argmax(q0, 0) + [x * m for x in range(n)]

        # store representatives in qr
        qr = qeqv[:, q0max_col_ind]
    else:
        if isinstance(samp_sym, str):
            qsym_s = quat_product_matrix(
                laue_group_to_quat(samp_sym), 'left'
            )  # sample symmetry operator
        else:
            qsym_s = quat_product_matrix(samp_sym, 'left')

        p = qsym_s.shape[0]  # number of sample symmetry operations

        # Do Gs * (R * Gc), store as
        # [Gs[:, 0:p]*q[:,   0]*Gc[:, 0], ..., Gs[:, 0:p]*q[:,   0]*Gc[:, m-1],
        #  ...,
        #  Gs[:, 0:p]*q[:, n-1]*Gc[:, 0], ..., Gs[:, 0:p]*q[:, n-1]*Gc[:, m-1]]
        qeqv = fix_quat(
            np.dot(qsym_s, qeqv).transpose(2, 0, 1).reshape(p * m * n, 4).T
        )

        raise NotImplementedError

    # debug
    assert qr.shape[1] == n, 'oops, something wrong here with your reshaping'

    if qdims == 3:
        qr = qr.T.reshape(l3, n3, 4).transpose(0, 2, 1)

    return qr


def laue_group_to_ltype(tag):
    """
    Yield lattice type of input tag.

    Parameters
    ----------
    tag : TYPE
        DESCRIPTION.

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    ltype : TYPE
        DESCRIPTION.

    """
    if not isinstance(tag, str):
        raise RuntimeError("entered flag is not a string!")

    if tag.lower() == 'ci' or tag.lower() == 's2':
        ltype = 'triclinic'
    elif tag.lower() == 'c2h':
        ltype = 'monoclinic'
    elif tag.lower() == 'd2h' or tag.lower() == 'vh':
        ltype = 'orthorhombic'
    elif tag.lower() == 'c4h' or tag.lower() == 'd4h':
        ltype = 'tetragonal'
    elif tag.lower() == 'c3i' or tag.lower() == 's6' or tag.lower() == 'd3d':
        ltype = 'trigonal'
    elif tag.lower() == 'c6h' or tag.lower() == 'd6h':
        ltype = 'hexagonal'
    elif tag.lower() == 'th' or tag.lower() == 'oh':
        ltype = 'cubic'
    else:
        raise RuntimeError(
            "unrecognized symmetry group.  "
            + "See ''help(laue_group_to_quat)'' for a list of valid options.  "
            + "Oh, and have a great day ;-)"
        )

    return ltype


def laue_group_to_quat(tag):
    """
    Return quaternion representation of requested symmetry group.

    Parameters
    ----------
    tag : str
        A case-insensitive string representing the Schoenflies symbol for the
        desired Laue group.  The 14 available choices are:

              Class           Symbol      N
             -------------------------------
              Triclinic       Ci (S2)     1
              Monoclinic      C2h         2
              Orthorhombic    D2h (Vh)    4
              Tetragonal      C4h         4
                              D4h         8
              Trigonal        C3i (S6)    3
                              D3d         6
              Hexagonal       C6h         6
                              D6h         12
              Cubic           Th          12
                              Oh          24

    Raises
    ------
    RuntimeError
        For invalid symmetry group tag.

    Returns
    -------
    qsym : (4, N) ndarray
        the quaterions associated with each element of the chosen symmetry
        group having n elements (dep. on group -- see INPUTS list above).

    Notes
    -----
    The conventions used for assigning a RHON basis, {x1, x2, x3}, to each
    point group are consistent with those published in Appendix B of [1]_.

    References
    ----------
    [1] Nye, J. F., ``Physical Properties of Crystals: Their
    Representation by Tensors and Matrices'', Oxford University Press,
    1985. ISBN 0198511655
    """
    if not isinstance(tag, str):
        raise RuntimeError("entered flag is not a string!")

    if tag.lower() == 'ci' or tag.lower() == 's2':
        # TRICLINIC
        angle_axis = np.vstack([0.0, 1.0, 0.0, 0.0])  # identity
    elif tag.lower() == 'c2h':
        # MONOCLINIC
        angle_axis = np.c_[
            [0.0, 1, 0, 0],  # identity
            [np.pi, 0, 1, 0],  # twofold about 010 (x2)
        ]
    elif tag.lower() == 'd2h' or tag.lower() == 'vh':
        # ORTHORHOMBIC
        angle_axis = np.c_[
            [0.0, 1, 0, 0],  # identity
            [np.pi, 1, 0, 0],  # twofold about 100
            [np.pi, 0, 1, 0],  # twofold about 010
            [np.pi, 0, 0, 1],  # twofold about 001
        ]
    elif tag.lower() == 'c4h':
        # TETRAGONAL (LOW)
        angle_axis = np.c_[
            [0.0, 1, 0, 0],  # identity
            [piby2, 0, 0, 1],  # fourfold about 001 (x3)
            [np.pi, 0, 0, 1],  #
            [piby2 * 3, 0, 0, 1],  #
        ]
    elif tag.lower() == 'd4h':
        # TETRAGONAL (HIGH)
        angle_axis = np.c_[
            [0.0, 1, 0, 0],  # identity
            [piby2, 0, 0, 1],  # fourfold about 0  0  1 (x3)
            [np.pi, 0, 0, 1],  #
            [piby2 * 3, 0, 0, 1],  #
            [np.pi, 1, 0, 0],  # twofold about  1  0  0 (x1)
            [np.pi, 0, 1, 0],  # twofold about  0  1  0 (x2)
            [np.pi, 1, 1, 0],  # twofold about  1  1  0
            [np.pi, -1, 1, 0],  # twofold about -1  1  0
        ]
    elif tag.lower() == 'c3i' or tag.lower() == 's6':
        # TRIGONAL (LOW)
        angle_axis = np.c_[
            [0.0, 1, 0, 0],  # identity
            [piby3 * 2, 0, 0, 1],  # threefold about 0001 (x3,c)
            [piby3 * 4, 0, 0, 1],  #
        ]
    elif tag.lower() == 'd3d':
        # TRIGONAL (HIGH)
        angle_axis = np.c_[
            [0.0, 1, 0, 0],  # identity
            [piby3 * 2, 0, 0, 1],  # threefold about 0001 (x3,c)
            [piby3 * 4, 0, 0, 1],  #
            [np.pi, 1, 0, 0],  # twofold about  2 -1 -1  0 (x1,a1)
            [np.pi, -0.5, sq3by2, 0],  # twofold about -1  2 -1  0 (a2)
            [np.pi, -0.5, -sq3by2, 0],  # twofold about -1 -1  2  0 (a3)
        ]
    elif tag.lower() == 'c6h':
        # HEXAGONAL (LOW)
        angle_axis = np.c_[
            [0.0, 1, 0, 0],  # identity
            [piby3, 0, 0, 1],  # sixfold about 0001 (x3,c)
            [piby3 * 2, 0, 0, 1],  #
            [np.pi, 0, 0, 1],  #
            [piby3 * 4, 0, 0, 1],  #
            [piby3 * 5, 0, 0, 1],  #
        ]
    elif tag.lower() == 'd6h':
        # HEXAGONAL (HIGH)
        angle_axis = np.c_[
            [0.0, 1, 0, 0],  # identity
            [piby3, 0, 0, 1],  # sixfold about  0  0  1 (x3,c)
            [piby3 * 2, 0, 0, 1],  #
            [np.pi, 0, 0, 1],  #
            [piby3 * 4, 0, 0, 1],  #
            [piby3 * 5, 0, 0, 1],  #
            [np.pi, 1, 0, 0],  # twofold about  2 -1  0 (x1,a1)
            [np.pi, -0.5, sq3by2, 0],  # twofold about -1  2  0 (a2)
            [np.pi, -0.5, -sq3by2, 0],  # twofold about -1 -1  0 (a3)
            [np.pi, sq3by2, 0.5, 0],  # twofold about  1  0  0
            [np.pi, 0, 1, 0],  # twofold about -1  1  0 (x2)
            [np.pi, -sq3by2, 0.5, 0],  # twofold about  0 -1  0
        ]
    elif tag.lower() == 'th':
        # CUBIC (LOW)
        angle_axis = np.c_[
            [0.0, 1, 0, 0],  # identity
            [np.pi, 1, 0, 0],  # twofold about    1  0  0 (x1)
            [np.pi, 0, 1, 0],  # twofold about    0  1  0 (x2)
            [np.pi, 0, 0, 1],  # twofold about    0  0  1 (x3)
            [piby3 * 2, 1, 1, 1],  # threefold about  1  1  1
            [piby3 * 4, 1, 1, 1],  #
            [piby3 * 2, -1, 1, 1],  # threefold about -1  1  1
            [piby3 * 4, -1, 1, 1],  #
            [piby3 * 2, -1, -1, 1],  # threefold about -1 -1  1
            [piby3 * 4, -1, -1, 1],  #
            [piby3 * 2, 1, -1, 1],  # threefold about  1 -1  1
            [piby3 * 4, 1, -1, 1],  #
        ]
    elif tag.lower() == 'oh':
        # CUBIC (HIGH)
        angle_axis = np.c_[
            [0.0, 1, 0, 0],  # identity
            [piby2, 1, 0, 0],  # fourfold about   1  0  0 (x1)
            [np.pi, 1, 0, 0],  #
            [piby2 * 3, 1, 0, 0],  #
            [piby2, 0, 1, 0],  # fourfold about   0  1  0 (x2)
            [np.pi, 0, 1, 0],  #
            [piby2 * 3, 0, 1, 0],  #
            [piby2, 0, 0, 1],  # fourfold about   0  0  1 (x3)
            [np.pi, 0, 0, 1],  #
            [piby2 * 3, 0, 0, 1],  #
            [piby3 * 2, 1, 1, 1],  # threefold about  1  1  1
            [piby3 * 4, 1, 1, 1],  #
            [piby3 * 2, -1, 1, 1],  # threefold about -1  1  1
            [piby3 * 4, -1, 1, 1],  #
            [piby3 * 2, -1, -1, 1],  # threefold about -1 -1  1
            [piby3 * 4, -1, -1, 1],  #
            [piby3 * 2, 1, -1, 1],  # threefold about  1 -1  1
            [piby3 * 4, 1, -1, 1],  #
            [np.pi, 1, 1, 0],  # twofold about    1  1  0
            [np.pi, -1, 1, 0],  # twofold about   -1  1  0
            [np.pi, 1, 0, 1],  # twofold about    1  0  1
            [np.pi, 0, 1, 1],  # twofold about    0  1  1
            [np.pi, -1, 0, 1],  # twofold about   -1  0  1
            [np.pi, 0, -1, 1],  # twofold about    0 -1  1
        ]
    else:
        raise RuntimeError(
            "unrecognized symmetry group.  "
            + "See ``help(laue_group_to_quat)'' for a list of valid options.  "
            + "Oh, and have a great day ;-)"
        )

    angle = angle_axis[0,]
    axis = angle_axis[1:,]

    #  Note: Axis does not need to be normalized in call to angle_axis_to_quat
    #  05/01/2014 JVB -- made output a contiguous C-ordered array
    qsym = np.array(angle_axis_to_quat(angle, axis).T, order='C').T

    return qsym
