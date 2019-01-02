import numpy as np

from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank


def tformfwd(trans, uv):
    """
        Function:
        ----------
            apply affine transform 'trans' to uv
        Parameters:
        ----------
            - trans: 3x3 np.array
                transform matrix
            - uv: Kx2 np.array
                each row is a pair of coordinates (x, y)
        Returns:
        ----------
            - xy: Kx2 np.array
                each row is a pair of transformed coordinates (x, y)
        """
    uv = np.hstack((
        uv, np.ones((uv.shape[0], 1))
    ))
    xy = np.dot(uv, trans)
    xy = xy[:, 0:-1]

    return xy


def tforminv(trans, uv):
    """
        Function:
        ----------
            apply the inverse of affine transform 'trans' to uv
        Parameters:
        ----------
            - trans: 3x3 np.array
                transform matrix
            - uv: Kx2 np.array
                each row is a pair of coordinates (x, y)
        Returns:
        ----------
            - xy: Kx2 np.array
                each row is a pair of inverse-transformed coordinates (x, y)
    """
    Tinv = inv(trans)
    xy = tformfwd(Tinv, uv)

    return xy


def find_nonreflective_similarity(uv, xy):
    """
        Function:
        ----------
            Find Non-reflective Similarity Transform Matrix 'trans'
        Parameters:
        ----------
            - uv: Kx2 np.array
                source points each row is a pair of coordinates (x, y)
            - xy: Kx2 np.array
                each row is a pair of inverse-transformed
        Returns:
            - trans: 3x3 np.array
                transform matrix from uv to xy
            - trans_inv: 3x3 np.array
                inverse of trans, transform matrix from xy to uv
    """
    options = {'K': 2}

    K = options['K']
    M = xy.shape[0]
    # use reshape to keep a column vector
    x = xy[:, 0].reshape((-1, 1))
    y = xy[:, 1].reshape((-1, 1))

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    # use reshape to keep a column vector
    u = uv[:, 0].reshape((-1, 1))
    v = uv[:, 1].reshape((-1, 1))
    U = np.vstack((u, v))

    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]
    t_inv = np.array([
        [sc, -ss, 0],
        [ss, sc, 0],
        [tx, ty, 1]
    ])
    t = inv(t_inv)
    t[:, 2] = np.array([0, 0, 1])

    return t, t_inv


def find_similarity(uv, xy):
    """
        Function:
        ----------
            Find Reflective Similarity Transform Matrix 'trans'
        Parameters:
        ----------
            - uv: Kx2 np.array
                source points each row is a pair of coordinates (x, y)
            - xy: Kx2 np.array
                each row is a pair of inverse-transformed
        Returns:
        ----------
            - trans: 3x3 np.array
                transform matrix from uv to xy
            - trans_inv: 3x3 np.array
                inverse of trans, transform matrix from xy to uv
    """
    options = {'K': 2}
    # Solve for trans1
    trans1, trans1_inv = find_nonreflective_similarity(uv, xy)
    # manually reflect the xy data across the Y-axis
    xyR = xy
    xyR[:, 0] = -1 * xyR[:, 0]
    trans2r, trans2r_inv = find_nonreflective_similarity(uv, xyR)
    # manually reflect the tform to undo the reflection done on xyR
    t_reflectY = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    trans2 = np.dot(trans2r, t_reflectY)
    # Figure out if trans1 or trans2 is better
    xy1 = tformfwd(trans1, uv)
    norm1 = norm(xy1 - xy)
    xy2 = tformfwd(trans2, uv)
    norm2 = norm(xy2 - xy)

    if norm1 <= norm2:
        return trans1, trans1_inv
    else:
        trans2_inv = inv(trans2)

    return trans2, trans2_inv


def get_similarity_transform(src_pts, dst_pts, reflective=True):
    """
        Function:
        ----------
            Find Similarity Transform Matrix 'trans'
        Parameters:
        ----------
            - src_pts: Kx2 np.array
                source points, each row is a pair of coordinates (x, y)
            - dst_pts: Kx2 np.array
                destination points, each row is a pair of transformed
                coordinates (x, y)
            - reflective: True or False
                if True:
                    use reflective similarity transform
                else:
                    use non-reflective similarity transform
        Returns:
        ----------
        - trans: 3x3 np.array
                transform matrix from uv to xy
            - trans_inv: 3x3 np.array
                inverse of trans, transform matrix from xy to uv
    """
    if reflective:
        trans, trans_inv = find_similarity(src_pts, dst_pts)
    else:
        trans, trans_inv = find_nonreflective_similarity(src_pts, dst_pts)

    return trans, trans_inv


def cvt_tform_mat_for_cv2(trans):
    """
        Function:
        ----------
            Convert Transform Matrix 'trans' into 'cv2_trans' which could be
            directly used by cv2.warpAffine()
        Parameters:
        ----------
            - trans: 3x3 np.array
                transform matrix from uv to xy
        Returns:
        ----------
            - cv2_trans: 2x3 np.array
                transform matrix from src_pts to dst_pts, could be directly 
                used for cv2.warpAffine()
    """
    cv2_trans = trans[:, 0:2].T

    return cv2_trans


def get_similarity_transform_for_cv2(src_pts, dst_pts, reflective=True):
    """
        Function:
        ----------
            Find Similarity Transform Matrix 'cv2_trans' which could be
            directly used by cv2.warpAffine()
        Parameters:
        ----------
            - src_pts: Kx2 np.array
                source points, each row is a pair of coordinates (x, y)
            - dst_pts: Kx2 np.array
                destination points, each row is a pair of transformed
                coordinates (x, y)
            reflective: True or False
                if True:
                    use reflective similarity transform
                else:
                    use non-reflective similarity transform
        Returns:
        ----------
            - cv2_trans: 2x3 np.array
                transform matrix from src_pts to dst_pts, could be directly
                used for cv2.warpAffine()
    """
    trans, trans_inv = get_similarity_transform(src_pts, dst_pts, reflective)
    cv2_trans = cvt_tform_mat_for_cv2(trans)

    return cv2_trans


def cpt2form(base, proj):
    """
        u = [0, 6, -2]
        v = [0, 3, 5]
        x = [-1, 0, 4]
        y = [-1, -10, 4]
        # In Matlab, run:
        #
        #   uv = [u'; v'];
        #   xy = [x'; y'];
        #   tform_sim=cp2tform(uv,xy,'similarity');
        #
        #   trans = tform_sim.tdata.T
        #   ans =
        #       -0.0764   -1.6190         0
        #        1.6190   -0.0764         0
        #       -3.2156    0.0290    1.0000
        #   trans_inv = tform_sim.tdata.Tinv
        #    ans =
        #
        #       -0.0291    0.6163         0
        #       -0.6163   -0.0291         0
        #       -0.0756    1.9826    1.0000
        #    xy_m=tformfwd(tform_sim, u,v)
        #
        #    xy_m =
        #
        #       -3.2156    0.0290
        #        1.1833   -9.9143
        #        5.0323    2.8853
        #    uv_m=tforminv(tform_sim, x,y)
        #
        #    uv_m =
        #
        #        0.5698    1.3953
        #        6.0872    2.2733
        #       -2.6570    4.3314
    """
    uv = np.squeeze(base)
    xy = np.squeeze(proj)
    trans, trans_inv = get_similarity_transform(uv, xy)

    return trans, trans_inv
