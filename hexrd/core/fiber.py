import numpy as np


from hexrd.core import constants as cnst
from hexrd.core.rotations import (
    quat_to_rot_mat,
    arccos_safe,
    angle_axis_to_quat,
    quat_product_matrix,
)
from hexrd.core.matrixutil import unit_vector, mult_mat_array, null_space


def distance_to_fiber(c, s, q, qsym, centrosymmetry=False, bmatrix=np.eye(3)):
    """
    Calculate symmetrically reduced distance to orientation fiber.

    Parameters
    ----------
    c : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.
    q : TYPE
        DESCRIPTION.
    qsym : TYPE
        DESCRIPTION.
    centrosymmetry : bool, optional
        If True, apply centrosymmetry to c. The default is False.
    bmatrix : np.ndarray, optional
        (3,3) b matrix. Default is the identity

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    d : TYPE
        DESCRIPTION.

    """
    if len(c) != 3 or len(s) != 3:
        raise RuntimeError('c and/or s are not 3-vectors')

    c = unit_vector(np.dot(bmatrix, np.asarray(c)))
    s = unit_vector(np.asarray(s).reshape(3, 1))

    nq = q.shape[1]  # number of quaternions
    rmats = quat_to_rot_mat(q)  # (nq, 3, 3)

    csym = apply_sym(c, qsym, centrosymmetry)  # (3, m)
    m = csym.shape[1]  # multiplicity

    if nq == 1:
        rc = np.dot(rmats, csym)  # apply q's to c's

        sdotrc = np.dot(s.T, rc).max()
    else:
        rc = mult_mat_array(
            rmats, np.tile(csym, (nq, 1, 1))
        )  # apply q's to c's

        sdotrc = (
            np.dot(s.T, rc.swapaxes(1, 2).reshape(nq * m, 3).T)
            .reshape(nq, m)
            .max(1)
        )

    d = arccos_safe(np.array(sdotrc))

    return d


def discrete_fiber(c, s, b_mat=np.eye(3), ndiv=120, csym=None, ssym=None):
    """
    Generate symmetrically reduced discrete orientation fiber.

    Parameters
    ----------
    c : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.
    B_MAT : TYPE, optional
        DESCRIPTION. The default is I3.
    ndiv : TYPE, optional
        DESCRIPTION. The default is 120.
    csym : TYPE, optional
        DESCRIPTION. The default is None.
    ssym : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    retval : TYPE
        DESCRIPTION.

    """

    ztol = cnst.SQRT_EPSF

    c = np.asarray(c).reshape((3, 1))
    s = np.asarray(s).reshape((3, 1))

    nptc = c.shape[1]
    npts = s.shape[1]

    c = unit_vector(
        np.dot(b_mat, c)
    )  # turn c hkls into unit vector in crys frame
    s = unit_vector(s)  # convert s to unit vector in samp frame

    retval = []
    for i_c in range(nptc):
        dupl_c = np.tile(c[:, i_c], (npts, 1)).T

        ax = s + dupl_c
        anrm = np.linalg.norm(ax, axis=0).squeeze()  # should be 1-d

        okay = anrm > ztol
        nokay = okay.sum()
        if nokay == npts:
            ax = ax / np.tile(anrm, (3, 1))
        else:
            nspace = null_space(c[:, i_c].reshape(3, 1))
            hperp = nspace[:, 0].reshape(3, 1)
            if nokay == 0:
                ax = np.tile(hperp, (1, npts))
            else:
                ax[:, okay] = ax[:, okay] / np.tile(anrm[okay], (3, 1))
                ax[:, not okay] = np.tile(hperp, (1, npts - nokay))

        q0 = np.vstack([np.zeros(npts), ax])

        # find rotations
        # note: the following line fixes bug with use of arange
        # with float increments
        phi = np.arange(0, ndiv) * (2 * np.pi / float(ndiv))
        qh = angle_axis_to_quat(phi, np.tile(c[:, i_c], (ndiv, 1)).T)

        # the fibers, arraged as (npts, 4, ndiv)
        qfib = np.dot(quat_product_matrix(qh, mult='right'), q0).transpose(
            2, 1, 0
        )
        if csym is not None:
            retval.append(
                to_fundamental_region(
                    qfib.squeeze(), crys_sym=csym, samp_sym=ssym
                )
            )
        else:
            retval.append(fix_quat(qfib).squeeze())
    return retval
