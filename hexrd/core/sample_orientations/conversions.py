import numpy as np
from numba import njit
from hexrd.core import constants

ap_2 = constants.SQUARE_HOMOCHORIC_RADIUS
sc = constants.SC


@njit(cache=True, nogil=True)
def get_pyramid(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    if (np.abs(x) <= z) and (np.abs(y) <= z):
        return 1

    elif (np.abs(x) <= -z) and (np.abs(y) <= -z):
        return 2

    elif (np.abs(z) <= x) and (np.abs(y) <= x):
        return 3

    elif (np.abs(z) <= -x) and (np.abs(y) <= -x):
        return 4

    elif (np.abs(x) <= y) and (np.abs(z) <= y):
        return 5

    elif (np.abs(x) <= -y) and (np.abs(z) <= -y):
        return 6


@njit(cache=True, nogil=True)
def cubochoric_to_rodrigues_fz(cu):
    ho = cubochoric_to_homochoric(cu)
    return homochoric_to_rodrigues_fz(ho)


@njit(cache=True, nogil=True)
def cubochoric_to_homochoric(cu):
    ma = np.max(np.abs(cu))
    assert ma <= ap_2, "point outside cubochoric grid"
    pyd = get_pyramid(cu)

    if pyd == 1 or pyd == 2:
        s_xyz = cu
    elif pyd == 3 or pyd == 4:
        s_xyz = np.array([cu[1], cu[2], cu[0]])
    elif pyd == 5 or pyd == 6:
        s_xyz = np.array([cu[2], cu[0], cu[1]])

    xyz = s_xyz * sc
    ma = np.max(np.abs(xyz))
    if ma < constants.SQRT_EPSF:
        return np.array([0.0, 0.0, 0.0])

    ma2 = np.max(np.abs(xyz[0:2]))
    if ma2 < constants.SQRT_EPSF:
        lam_xyz = np.array([0.0, 0.0, constants.PREF * xyz[2]])

    else:
        if np.abs(xyz[1]) <= np.abs(xyz[0]):
            q = (np.pi/12.0) * xyz[1]/xyz[0]
            c = np.cos(q)
            s = np.sin(q)
            q = constants.PREK * xyz[0] / np.sqrt(np.sqrt(2.0)-c)
            t1 = (np.sqrt(2.0) * c - 1.0) * q
            t2 = np.sqrt(2.0) * s * q
        else:
            q = (np.pi/12.0) * xyz[0]/xyz[1]
            c = np.cos(q)
            s = np.sin(q)
            q = constants.PREK * xyz[1] / np.sqrt(np.sqrt(2.0)-c)
            t1 = np.sqrt(2.0) * s * q
            t2 = (np.sqrt(2.0) * c - 1.0) * q

        c = t1**2 + t2**2
        s = np.pi * c / (24.0 * xyz[2]**2)
        c = np.sqrt(np.pi) * c / np.sqrt(24.0) / xyz[2]
        q = np.sqrt( 1.0 - s )
        lam_xyz = np.array([t1 * q, t2 * q, constants.PREF * xyz[2] - c])

    if pyd in (1, 2):
        return lam_xyz
    if pyd in (3, 4):
        return np.array([lam_xyz[2], lam_xyz[0], lam_xyz[1]])
    # pyd in (4, 5)
    return np.array([lam_xyz[1], lam_xyz[2], lam_xyz[0]])


@njit(cache=True, nogil=True)
def homochoric_to_rodrigues_fz(ho):
    ax = homochoric_to_axis_angle(ho)
    return axis_angle_to_rodrigues_fz(ax)


@njit(cache=True, nogil=True)
def homochoric_to_axis_angle(ho):
    hmag = np.linalg.norm(ho[:])**2
    if hmag < constants.SQRT_EPSF:
        return np.array([0.0, 0.0, 1.0, 0.0])
    hm = hmag
    hn = ho/np.sqrt(hmag)
    s = constants.T_FIT[0] + constants.T_FIT[1] * hmag
    for ii in range(2, 21):
        hm = hm*hmag
        s = s + constants.T_FIT[ii] * hm
    s = 2.0 * np.arccos(s)
    diff = np.abs(s - np.pi)
    if diff < constants.SQRT_EPSF:
        return np.array([hn[0], hn[1], hn[2], np.pi])
    else:
        return np.array([hn[0], hn[1], hn[2], s])


@njit(cache=True, nogil=True)
def axis_angle_to_rodrigues_fz(ax):
    if np.abs(ax[3]) < constants.SQRT_EPSF:
        return np.array([0.0, 0.0, 1.0, 0.0])

    elif np.abs(ax[3] - np.pi) < constants.SQRT_EPSF:
        return np.array([ax[0], ax[1], ax[2], np.inf])

    else:
        return np.array([ax[0], ax[1], ax[2], np.tan(ax[3]*0.5)])


@njit(cache=True, nogil=True)
def rodrigues_fz_to_quaternion(ro):
    ax = rodrigues_fz_to_axis_angle(ro)
    return axis_angle_to_quaternion(ax)


@njit(cache=True, nogil=True)
def rodrigues_fz_to_axis_angle(ro):
    if np.abs(ro[3]) < constants.SQRT_EPSF:
        return np.array([0.0, 0.0, 1.0, 0.0])
    elif ro[3] == np.inf:
        return np.array([ro[0], ro[1], ro[2], np.pi])
    else:
        ang = 2.0*np.arctan(ro[3])
        mag = 1.0/np.linalg.norm(ro[0:3])
        return np.array([ro[0]*mag, ro[1]*mag, ro[2]*mag, ang])


@njit(cache=True, nogil=True)
def axis_angle_to_quaternion(ax):
    if np.abs(ax[3]) < constants.SQRT_EPSF:
        return np.array([1.0, 0.0, 0.0, 0.0])
    else:
        c = np.cos(ax[3]*0.5)
        s = np.sin(ax[3]*0.5)
        return np.array([c, ax[0]*s, ax[1]*s, ax[2]*s])
