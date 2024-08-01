# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (c) 2012, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by Joel Bernier <bernier2@llnl.gov> and others.
# LLNL-CODE-529294.
# All rights reserved.
#
# This file is part of HEXRD. For details on dowloading the source,
# see the file COPYING.
#
# Please also see the file LICENSE.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program (see file LICENSE); if not, write to
# the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
# Boston, MA 02111-1307 USA or visit <http://www.gnu.org/licenses/>.
# =============================================================================

import numpy as np

from numpy.linalg import svd

from scipy import sparse
import numba


from hexrd.core import constants

# module variables
sqr6i = 1.0 / np.sqrt(6.0)
sqr3i = 1.0 / np.sqrt(3.0)
sqr2i = 1.0 / np.sqrt(2.0)
sqr2 = np.sqrt(2.0)
sqr3 = np.sqrt(3.0)
sqr2b3 = np.sqrt(2.0 / 3.0)

tolerance = constants.EPSF
v_tol = 100 * constants.EPSF


def unit_vector(a):
    """
    normalize array of column vectors (hstacked, axis = 0)
    """
    assert a.ndim in (
        1,
        2,
    ), f"incorrect arg shape; must be 1-d or 2-d, yours is {a.ndim}-d"
    a = np.asarray(a)

    norms = np.atleast_1d(np.linalg.norm(a, axis=0))

    # prevent divide by zero
    norms[norms <= constants.TEN_EPSF] = 1.0
    return a / norms


def null_space(matrix, tol=v_tol):
    """
    computes the null space of the real matrix
    """
    assert matrix.ndim == 2, f"input must be 2-d; yours is {matrix.ndim}-d"

    n, m = matrix.shape

    if n > m:
        return null_space(matrix.T, tol).T

    _, s, v = svd(matrix)

    s = np.hstack([s, np.zeros(m - n)])

    null_mask = s <= tol

    return v[null_mask, :]


def mat_array_to_block_sparce(mat_array):
    """
    mat_array_to_block_sparce

    Constructs a block diagonal sparse matrix (csc format) from a
    (p, m, n) ndarray of p (m, n) arrays

    ...maybe optional args to pick format type?
    """

    # if isinstance(args[0], str):
    #    a = args[0]
    # if a == 'csc': ...

    if len(mat_array.shape) != 3:
        raise RuntimeError("input array is not the correct shape!")

    p, m, n = mat_array.shape

    jmax = p * n
    imax = p * m
    ntot = p * m * n

    rl = np.arange(p)
    rm = np.arange(m)
    rjmax = np.arange(jmax)

    sij = mat_array.transpose(0, 2, 1).reshape(1, ntot).squeeze()
    j = np.reshape(np.tile(rjmax, (m, 1)).T, (1, ntot))
    i = np.reshape(np.tile(rm, (1, jmax)), (1, ntot)) + np.reshape(
        np.tile(m * rl, (m * n, 1)).T, (1, ntot)
    )

    ij = np.concatenate((i, j), axis=0)

    # syntax as of scipy-0.7.0
    # csc_matrix((data, indices, indptr), shape=(M, N))
    smat = sparse.csc_matrix((sij, ij), shape=(imax, jmax))

    return smat


def symm_to_vec_mv(matrix, scale=True):
    """
    convert from symmetric matrix to Mandel-Voigt vector
    representation (JVB)
    """
    if scale:
        fac = sqr2
    else:
        fac = 1.0
    mvvec = np.zeros(6, dtype='float64')
    mvvec[0] = matrix[0, 0]
    mvvec[1] = matrix[1, 1]
    mvvec[2] = matrix[2, 2]
    mvvec[3] = fac * matrix[1, 2]
    mvvec[4] = fac * matrix[0, 2]
    mvvec[5] = fac * matrix[0, 1]
    return mvvec


def vec_mv_to_symm(vector, scale=True):
    """
    convert from Mandel-Voigt vector to symmetric matrix
    representation (JVB)
    """
    if scale:
        fac = sqr2
    else:
        fac = 1.0
    symm_mat = np.zeros((3, 3), dtype='float64')
    symm_mat[0, 0] = vector[0]
    symm_mat[1, 1] = vector[1]
    symm_mat[2, 2] = vector[2]
    symm_mat[1, 2] = vector[3] / fac
    symm_mat[0, 2] = vector[4] / fac
    symm_mat[0, 1] = vector[5] / fac
    symm_mat[2, 1] = vector[3] / fac
    symm_mat[2, 0] = vector[4] / fac
    symm_mat[1, 0] = vector[5] / fac
    return symm_mat


def vec_mv_cob_matrix(rot):
    """
    GenerateS array of 6 x 6 basis transformation matrices for the
    Mandel-Voigt tensor representation in 3-D given by:

    [A] = [[A_11, A_12, A_13],
           [A_12, A_22, A_23],
           [A_13, A_23, A_33]]

    {A} = [A_11, A_22, A_33, sqrt(2)*A_23, sqrt(2)*A_13, sqrt(2)*A_12]

    where the operation :math:`R*A*R.T` (in tensor notation) is obtained by
    the matrix-vector product [T]*{A}.

    USAGE

        T = vec_mv_cob_matrix(rot)

    INPUTS

        1) rot is (3, 3) an ndarray representing a change of basis matrix

    OUTPUTS

        1) T is (6, 6), an ndarray of transformation matrices as
           described above

    NOTES

        1) Compoments of symmetric 4th-rank tensors transform in a
           manner analogous to symmetric 2nd-rank tensors in full
           matrix notation.

    SEE ALSO

    symmToVecMV, vecMVToSymm, quatToMat
    """
    rdim = len(rot.shape)
    if rdim == 2:
        nrot = 1
        rot = np.tile(rot, (1, 1, 1))
    elif rdim == 3:
        nrot = rot.shape[0]
    else:
        raise RuntimeError(
            "rot array must be (3, 3) or (n, 3, 3); "
            f"input has dimension {rdim}"
        )

    cob_mat = np.zeros((nrot, 6, 6), dtype='float64')

    for i in range(3):
        # Other two i values
        i1, i2 = [k for k in range(3) if k != i]
        for j in range(3):
            # Other two j values
            j1, j2 = [k for k in range(3) if k != j]

            cob_mat[:, i, j] = rot[:, i, j] ** 2
            cob_mat[:, i, j + 3] = sqr2 * rot[:, i, j1] * rot[:, i, j2]
            cob_mat[:, i + 3, j] = sqr2 * rot[:, i1, j] * rot[:, i2, j]
            cob_mat[:, i + 3, j + 3] = (
                rot[:, i1, j1] * rot[:, i2, j2]
                + rot[:, i1, j2] * rot[:, i2, j1]
            )

    return cob_mat.squeeze()


def vec_mv_to_normal_projection(vec):
    """
    Gives vstacked p x 6 array to To perform n' * A * n as [N]*{A} for
    p hstacked input 3-vectors using the Mandel-Voigt convention.

    Nvec = vec_mv_to_normal_projection(vec)

    *) the input vector array need not be normalized; it is performed in place

    """
    # normalize in place... col vectors!
    n = unit_vector(vec)

    nmat = np.array(
        [
            n[0, :] ** 2,
            n[1, :] ** 2,
            n[2, :] ** 2,
            sqr2 * n[1, :] * n[2, :],
            sqr2 * n[0, :] * n[2, :],
            sqr2 * n[0, :] * n[1, :],
        ],
        dtype='float64',
    )

    return nmat.T


def rank_one_matrix(vec1, vec2=None):
    """
    Create rank one matrices (dyadics) from vectors.

      r1mat = rank_one_matrix(vec1)
      r1mat = rank_one_matrix(vec1, vec2)

      vec1 is m1 x n, an array of n hstacked m1 vectors
      vec2 is m2 x n, (optional) another array of n hstacked m2 vectors

      r1mat is n x m1 x m2, an array of n rank one matrices
                   formed as c1*c2' from columns c1 and c2

      With one argument, the second vector is taken to
      the same as the first.

      Notes:

      *)  This routine loops on the dimension m, assuming this
          is much smaller than the number of points, n.
    """
    if len(vec1.shape) > 2:
        raise RuntimeError("input vec1 is the wrong shape")

    if vec2 is None:
        vec2 = vec1.copy()
    elif len(vec2.shape) > 2:
        raise RuntimeError("input vec2 is the wrong shape")

    m1, n1 = np.atleast_2d(vec1).shape
    m2, n2 = np.atleast_2d(vec2).shape

    if n1 != n2:
        raise RuntimeError("Number of vectors differ in arguments.")

    m1m2 = m1 * m2

    r1mat = np.zeros((m1m2, n1), dtype='float64')

    mrange = np.arange(m1)

    for i in range(m2):
        r1mat[mrange, :] = vec1 * np.tile(vec2[i, :], (m1, 1))
        mrange = mrange + m1

    r1mat = np.reshape(r1mat.T, (n1, m2, m1)).transpose(0, 2, 1)
    return r1mat.squeeze()


def skew(rot):
    """
    skew-symmetric decomposition of n square (m, m) ndarrays.  Result
    is a (squeezed) (n, m, m) ndarray
    """
    rot = np.asarray(rot)

    if rot.ndim == 2:
        m = rot.shape[0]
        n = rot.shape[1]
        if m != n:
            raise RuntimeError(
                "this function only works for square arrays; "
                f"yours is ({m}, {n})"
            )
        rot.resize(1, m, n)
    elif rot.ndim == 3:
        m = rot.shape[1]
        n = rot.shape[2]
        if m != n:
            raise RuntimeError("this function only works for square arrays")
    else:
        raise RuntimeError("this function only works for square arrays")

    return np.squeeze(0.5 * (rot - rot.transpose(0, 2, 1)))


def symm(rot):
    """
    symmetric decomposition of n square (m, m) ndarrays.  Result
    is a (squeezed) (n, m, m) ndarray.
    """
    rot = np.asarray(rot)

    if rot.ndim == 2:
        m = rot.shape[0]
        n = rot.shape[1]
        if m != n:
            raise RuntimeError(
                "this function only works for square arrays; "
                f"yours is ({m}, {n})"
            )
        rot.resize(1, m, n)
    elif rot.ndim == 3:
        m = rot.shape[1]
        n = rot.shape[2]
        if m != n:
            raise RuntimeError("this function only works for square arrays")
    else:
        raise RuntimeError("this function only works for square arrays")

    return np.squeeze(0.5 * (rot + rot.transpose(0, 2, 1)))


def vector_to_skew_matrix(w):
    """
    vector_to_skew_matrix(w)

    given a (3, n) ndarray, w,  of n hstacked axial vectors, computes
    the associated skew matrices and stores them in an (n, 3, 3)
    ndarray.  Result is (3, 3) for w.shape = (3, 1) or (3, ).

    See also: skew_matrix_to_vector
    """
    dims = w.ndim
    stackdim = 0
    if dims == 1:
        if len(w) != 3:
            raise RuntimeError('input is not a 3-d vector')
        w = np.vstack(w)
        stackdim = 1
    elif dims == 2:
        if w.shape[0] != 3:
            raise RuntimeError(
                'input is of incorrect shape; expecting shape[0] = 3'
            )
        stackdim = w.shape[1]
    else:
        raise RuntimeError('input is incorrect shape; expecting ndim = 1 or 2')

    zs = np.zeros((1, stackdim), dtype='float64')
    skew_mat = np.vstack(
        [zs, -w[2, :], w[1, :], w[2, :], zs, -w[0, :], -w[1, :], w[0, :], zs]
    )

    return np.squeeze(np.reshape(skew_mat.T, (stackdim, 3, 3)))


def skew_matrix_to_vector(skew_mat):
    """
    skew_matrix_to_vector(W)

    given an (n, 3, 3) or (3, 3) ndarray, W, of n stacked 3x3 skew
    matrices, computes the associated axial vector(s) and stores them
    in an (3, n) ndarray.  Result always has ndim = 2.

    See also: vector_to_skew_matrix
    """
    stackdim = 0
    if skew_mat.ndim == 2:
        if skew_mat.shape[0] != 3 or skew_mat.shape[0] != 3:
            raise RuntimeError('input is not (3, 3)')
        stackdim = 1
        skew_mat.resize(1, 3, 3)
    elif skew_mat.ndim == 3:
        if skew_mat.shape[1] != 3 or skew_mat.shape[2] != 3:
            raise RuntimeError('input is not (3, 3)')
        stackdim = skew_mat.shape[0]
    else:
        raise RuntimeError('input is incorrect shape; expecting (n, 3, 3)')

    w = np.zeros((3, stackdim), dtype='float64')
    for i in range(stackdim):
        w[:, i] = np.r_[
            -skew_mat[i, 1, 2], skew_mat[i, 0, 2], -skew_mat[i, 0, 1]
        ]

    return w


def mult_mat_array(ma1, ma2):
    """
    multiply two 3-d arrays of 2-d matrices
    """
    shp1 = ma1.shape
    shp2 = ma2.shape

    if len(shp1) != 3 or len(shp2) != 3:
        raise RuntimeError(
            'input is incorrect shape; '
            + 'expecting len(ma1).shape = len(ma2).shape = 3'
        )

    if shp1[0] != shp2[0]:
        raise RuntimeError('mismatch on number of matrices')

    if shp1[2] != shp2[1]:
        raise RuntimeError('mismatch on internal matrix dimensions')

    prod = np.zeros((shp1[0], shp1[1], shp2[2]))
    for j in range(shp1[0]):
        prod[j, :, :] = np.dot(ma1[j, :, :], ma2[j, :, :])

    return prod


def unique_vectors(v, tol=1.0e-12):
    """
    Sort vectors and discard duplicates.

      USAGE:

          uvec = unique_vectors(vec, tol=1.0e-12)

    v   --
    tol -- (optional) comparison tolerance

    D. E. Boyce 2010-03-18
    """

    vdims = v.shape

    iv = np.zeros(vdims)
    for row in range(vdims[0]):
        tmpord = np.argsort(v[row, :]).tolist()
        tmpsrt = v[np.ix_([row], tmpord)].squeeze()
        tmpcmp = abs(tmpsrt[1:] - tmpsrt[0:-1])
        indep = np.hstack([True, tmpcmp > tol])  # independent values
        rowint = indep.cumsum()
        iv[np.ix_([row], tmpord)] = rowint
    #
    #  Dictionary sort from bottom up
    #
    i_num = np.lexsort(iv)
    iv_srt = iv[:, i_num]
    v_srt = v[:, i_num]

    iv_ind = np.zeros(vdims[1], dtype='int')
    n_uniq = 1
    iv_ind[0] = 0
    for col in range(1, vdims[1]):
        if any(iv_srt[:, col] != iv_srt[:, col - 1]):
            iv_ind[n_uniq] = col
            n_uniq += 1

    return v_srt[:, iv_ind[0:n_uniq]]


def find_duplicate_vectors(vec, tol=v_tol, ignore_sign=False):
    """
    Find vectors in an array that are equivalent to within
    a specified tolerance

      USAGE:

          eqv = DuplicateVectors(vec, *tol)

      INPUT:

          1) vec is n x m, a double array of m horizontally concatenated
                           n-dimensional vectors.
         *2) tol is 1 x 1, a scalar tolerance.  If not specified, the default
                           tolerance is 1e-14.
         *3) set ignore_sign to True if vec and -vec
             are to be treated as equivalent

      OUTPUT:

          1) eqv is 1 x p, a list of p equivalence relationships.

      NOTES:

          Each equivalence relationship is a 1 x q vector of indices that
          represent the locations of duplicate columns/entries in the array
          vec.  For example:

                | 1     2     2     2     1     2     7 |
          vec = |                                       |
                | 2     3     5     3     2     3     3 |

          eqv = [[1x2 double]    [1x3 double]], where

          eqv[0] = [0  4]
          eqv[1] = [1  3  5]
    """
    eqv = _find_duplicate_vectors(vec, tol, ignore_sign)
    uid = np.arange(0, vec.shape[1], dtype=np.int64)
    mask = ~np.isnan(eqv)
    idx = eqv[mask].astype(np.int64)
    uid2 = list(np.delete(uid, idx))
    eqv2 = []
    for ii in range(eqv.shape[0]):
        v = eqv[ii, mask[ii, :]]
        if v.shape[0] > 0:
            eqv2.append([ii] + list(v.astype(np.int64)))
    return eqv2, uid2


@numba.njit(cache=True, nogil=True)
def _find_duplicate_vectors(vec, tol, ignore_sign):
    if ignore_sign:
        vec2 = -vec.copy()
    m = vec.shape[1]

    eqv = np.zeros((m, m), dtype=np.float64)
    eqv[:] = np.nan
    eqv_elem_master = []

    for ii in range(m):
        ctr = 0
        eqv_elem = np.zeros((m,), dtype=np.int64)
        for jj in range(ii + 1, m):
            if not jj in eqv_elem_master:
                if ignore_sign:
                    diff = np.sum(np.abs(vec[:, ii] - vec2[:, jj]))
                    diff2 = np.sum(np.abs(vec[:, ii] - vec[:, jj]))
                    if diff < tol or diff2 < tol:
                        eqv_elem[ctr] = jj
                        eqv_elem_master.append(jj)
                        ctr += 1
                else:
                    diff = np.sum(np.abs(vec[:, ii] - vec[:, jj]))
                    if diff < tol:
                        eqv_elem[ctr] = jj
                        eqv_elem_master.append(jj)
                        ctr += 1

        for kk in range(ctr):
            eqv[ii, kk] = eqv_elem[kk]

    return eqv


def strain_ten_to_vec(strain_ten):
    strain_vec = np.zeros(6, dtype='float64')
    strain_vec[0] = strain_ten[0, 0]
    strain_vec[1] = strain_ten[1, 1]
    strain_vec[2] = strain_ten[2, 2]
    strain_vec[3] = 2 * strain_ten[1, 2]
    strain_vec[4] = 2 * strain_ten[0, 2]
    strain_vec[5] = 2 * strain_ten[0, 1]
    strain_vec = np.atleast_2d(strain_vec).T
    return strain_vec


def strain_vec_to_ten(strain_vec):
    strain_ten = np.zeros((3, 3), dtype='float64')
    strain_ten[0, 0] = strain_vec[0]
    strain_ten[1, 1] = strain_vec[1]
    strain_ten[2, 2] = strain_vec[2]
    strain_ten[1, 2] = strain_vec[3] / 2.0
    strain_ten[0, 2] = strain_vec[4] / 2.0
    strain_ten[0, 1] = strain_vec[5] / 2.0
    strain_ten[2, 1] = strain_vec[3] / 2.0
    strain_ten[2, 0] = strain_vec[4] / 2.0
    strain_ten[1, 0] = strain_vec[5] / 2.0
    return strain_ten


def stress_ten_to_vec(stress_ten):
    stress_vec = np.zeros(6, dtype='float64')
    stress_vec[0] = stress_ten[0, 0]
    stress_vec[1] = stress_ten[1, 1]
    stress_vec[2] = stress_ten[2, 2]
    stress_vec[3] = stress_ten[1, 2]
    stress_vec[4] = stress_ten[0, 2]
    stress_vec[5] = stress_ten[0, 1]
    stress_vec = np.atleast_2d(stress_vec).T
    return stress_vec


def stress_vec_to_ten(stress_vec):

    stress_ten = np.zeros((3, 3), dtype='float64')
    stress_ten[0, 0] = stress_vec[0]
    stress_ten[1, 1] = stress_vec[1]
    stress_ten[2, 2] = stress_vec[2]
    stress_ten[1, 2] = stress_vec[3]
    stress_ten[0, 2] = stress_vec[4]
    stress_ten[0, 1] = stress_vec[5]
    stress_ten[2, 1] = stress_vec[3]
    stress_ten[2, 0] = stress_vec[4]
    stress_ten[1, 0] = stress_vec[5]

    return stress_ten


def ale_3d_strain_out_to_symm_mat(vecds):
    """
    convert from vecds representation to symmetry matrix
    takes 5 components of evecd and the 6th component is lndetv
    """
    eps = np.zeros([3, 3], dtype='float64')

    a = np.exp(vecds[5]) ** (1.0 / 3.0)  # -p
    t1 = sqr2i * vecds[0]
    t2 = sqr6i * vecds[1]

    eps[0, 0] = t1 - t2
    eps[1, 1] = -t1 - t2
    eps[2, 2] = sqr2b3 * vecds[1]
    eps[1, 0] = vecds[2] * sqr2i
    eps[2, 0] = vecds[3] * sqr2i
    eps[2, 1] = vecds[4] * sqr2i

    eps[0, 1] = eps[1, 0]
    eps[0, 2] = eps[2, 0]
    eps[1, 2] = eps[2, 1]

    epstar = eps / a

    v = (np.eye(3) + epstar) * a
    v_inv = (np.eye(3) - epstar) / a

    return v, v_inv


def vecds_to_symm_mat(vecds):
    """convert from vecds representation to symmetry matrix"""
    sym = np.zeros([3, 3], dtype='float64')
    akk_by_3 = sqr3i * vecds[5]  # -p
    t1 = sqr2i * vecds[0]
    t2 = sqr6i * vecds[1]

    sym[0, 0] = t1 - t2 + akk_by_3
    sym[1, 1] = -t1 - t2 + akk_by_3
    sym[2, 2] = sqr2b3 * vecds[1] + akk_by_3
    sym[1, 0] = vecds[2] * sqr2i
    sym[2, 0] = vecds[3] * sqr2i
    sym[2, 1] = vecds[4] * sqr2i

    sym[0, 1] = sym[1, 0]
    sym[0, 2] = sym[2, 0]
    sym[1, 2] = sym[2, 1]
    return sym


def trace_to_vecds_s(akk):
    return sqr3i * akk


def vecds_s_to_trace(vecds_s):
    return vecds_s * sqr3


def symm_to_vecds(sym):
    """convert from symmetry matrix to vecds representation"""
    vecds = np.zeros(6, dtype='float64')
    vecds[0] = sqr2i * (sym[0, 0] - sym[1, 1])
    vecds[1] = sqr6i * (2.0 * sym[2, 2] - sym[0, 0] - sym[1, 1])
    vecds[2] = sqr2 * sym[1, 0]
    vecds[3] = sqr2 * sym[2, 0]
    vecds[4] = sqr2 * sym[2, 1]
    vecds[5] = trace_to_vecds_s(np.trace(sym))
    return vecds


def solve_wahba(v, w, weights=None):
    """
    take unique vectors 3-vectors v = [[v0], [v1], ..., [vn]] in frame 1 that
    are aligned with vectors w = [[w0], [w1], ..., [wn]] in frame 2 and solve
    for the rotation that takes components in frame 1 to frame 2

    minimizes the cost function:

      J(R) = 0.5 * sum_{k=1}^{N} a_k * || w_k - R*v_k ||^2

    INPUTS:
      v is list-like, where each entry is a length 3 vector
      w is list-like, where each entry is a length 3 vector

      len(v) == len(w)

      weights are optional, and must have the same length as v, w

    OUTPUT:
      (3, 3) orthognal matrix that takes components in frame 1 to frame 2
    """
    n_vecs = len(v)

    assert len(w) == n_vecs

    if weights is not None:
        assert len(weights) == n_vecs
    else:
        weights = np.ones(n_vecs)

    # cast v, w, as arrays if not
    v = np.atleast_2d(v)
    w = np.atleast_2d(w)

    # compute weighted outer product sum
    weighted_ops = np.zeros((3, 3))
    for i in range(n_vecs):
        weighted_ops += weights[i] * np.dot(
            w[i].reshape(3, 1), v[i].reshape(1, 3)
        )

    # compute svd
    us, _, vs_t = svd(weighted_ops)

    # form diagonal matrix for solution
    m = np.diag([1.0, 1.0, np.linalg.det(us) * np.linalg.det(vs_t)])
    return np.dot(us, np.dot(m, vs_t))


# =============================================================================
# Numba-fied frame cache writer
# =============================================================================


@numba.njit(cache=True, nogil=True)
def extract_ijv(in_array, threshold, out_i, out_j, out_v):
    n = 0
    w, h = in_array.shape
    for i in range(w):
        for j in range(h):
            v = in_array[i, j]
            if v > threshold:
                out_i[n] = i
                out_j[n] = j
                out_v[n] = v
                n += 1
    return n
