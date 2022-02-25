import cv2 as cv
import numpy as np
from scipy.interpolate import griddata
from scipy.linalg import svd
from scipy.optimize import least_squares

from scipy.spatial.transform import Rotation


def detect_keypoints(images, pattern_size, edge_length=1.0, invert=False):
    """
    :param edge_length: The length of the edge of a single quad in meters
    """
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    num_points = pattern_size[0] * pattern_size[1]

    # Points in the board's coordinate frame
    scene_points = np.zeros((num_points, 3), np.float32)
    scene_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * edge_length

    detections = {}

    for key, img in images.items():
        # Ordinary image with 0 - 255
        if len(img.shape) == 3 and img.shape[-1] == 3:
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Infra-red image
        elif len(img.shape) == 2:
            gray_img = img

        else:
            raise NotImplementedError

        success, kp = cv.findChessboardCorners(gray_img, pattern_size,
                                               flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK)

        if success:
            kp = cv.cornerSubPix(gray_img, kp, (11, 11), (-1, -1), criteria)

            if invert:
                kp = kp[::-1, :, :]

            detections[key] = (scene_points, kp)

    return detections


def undistort_intrinsics(shape, intrinsics, dist_coeff):
    undist_intrinsics, _ = cv.getOptimalNewCameraMatrix(intrinsics, dist_coeff, shape, 1, shape)

    return undist_intrinsics


def undistort_images(images, intrinsics, dist_coeff, undist_intrinsics, inter_method):
    undist_images = {}

    for key, image in images.items():
        if len(image.shape) == 3:
            shape = image.shape[::-1][1:]

        elif len(image.shape) == 2:
            shape = image.shape[::-1]

        else:
            raise NotImplementedError

        map_x, map_y = cv.initUndistortRectifyMap(intrinsics, dist_coeff, None, undist_intrinsics, shape, cv.CV_32FC1)
        undist_image = cv.remap(image, map_x, map_y, inter_method)

        undist_images[key] = undist_image

    return undist_images


def pointcloudify_depths(depths, undist_intrinsics):
    pcd_depths = {}

    for key, depth in depths.items():
        shape = depth.shape[::-1]

        grid_x, grid_y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        grid = np.concatenate([np.expand_dims(grid_x, -1),
                               np.expand_dims(grid_y, -1)], axis=-1)

        norm_grid = to_norm_image_coord(grid.reshape(-1, 2), undist_intrinsics)

        # Raise by undistorted depth value from image plane to local camera space
        local_grid = norm_grid * np.expand_dims(depth.reshape(-1), axis=-1)

        pcd_depths[key] = local_grid.astype(np.float32)

    return pcd_depths


def plane2plane_transformation(plane_points1, plane_points2):
    """
    :param plane_points1: N x 3
    :param plane_points2: N x 3
    """
    centroid1, centroid2 = plane_points1.mean(axis=0), plane_points2.mean(axis=0)

    plane_points1 = plane_points1 - centroid1.reshape(1, 3)
    plane_points2 = plane_points2 - centroid2.reshape(1, 3)

    H = plane_points2.T.dot(plane_points1).T

    u, d, vt = svd(H, full_matrices=False)

    # If a singular value is close to zero then the general solution will not hold
    num_sing = sum(d < 1e-10)
    if num_sing == 1:
        w, v = np.linalg.eig(H.T @ H)

        sing_mask = w > 1e-10
        s = sum([np.outer(vi, vi) / np.sqrt(wi) for wi, vi in zip(w[sing_mask], v[:, sing_mask].T)])
        s3 = np.outer(v[~sing_mask], v[~sing_mask])

        R = H.T @ s + s3

        if np.linalg.det(R) < 0:
            R = H.T @ s - s3

    elif num_sing == 0:
        R = u @ vt
    else:
        raise RuntimeError

    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = centroid2 - R @ centroid1
    T[3, 3] = 1

    return T


def compose_fund_mat(T, K1, K2):
    E = compose_ess_mat(T)

    iK1, iK2 = np.linalg.inv(K1), np.linalg.inv(K2)

    return iK2.T @ E @ iK1


def optimize_pose_lm(T, kp1, kp2, K1, K2, batch=False):
    """
    :param T: 4 x 4
    :param kp1: list of N x 2 or N x 2
    :param kp2: list of N x 2 or N x 2
    :param K1: list of 3 x 3 or 3 x 3
    :param K2: list of 3 x 3 or 3 x 3
    """
    T = T.copy()
    dist = np.linalg.norm(T[:3, 3])
    T[:3, 3] /= dist

    x = encode_essmat(T)

    iK1 = np.linalg.inv(K1)
    iK2 = np.linalg.inv(K2)

    lm_opt = least_squares(loss_fun_pose, x, jac=loss_fun_pose_jac, args=(kp1, kp2, iK1, iK2), method='lm')

    if lm_opt.success:
        avg_ep_dist = np.abs(lm_opt.fun).mean()
        print("Avg. epipolar distance:", avg_ep_dist)
        print("Number of iters:", lm_opt.nfev)

        T = np.zeros((4, 4))
        T[:3, :3] = cv.Rodrigues(lm_opt.x[:3])[0]

        t = np.zeros(3)
        t[0:2] = lm_opt.x[3:]
        t = cv.Rodrigues(t)[0][:, 2]
        T[:3, 3] = t * dist

        T[3, 3] = 1

        return T

    else:
        return None


def optimize_translation_lm(T, local_kp1, kp2, K2):
    m = 1

    lm_opt = least_squares(loss_fun_tr, m, args=(T, local_kp1, kp2, K2), method='lm')

    if lm_opt.success:
        avg_norm = np.abs(lm_opt.fun).mean()

        print("Avg. l2-norm:", avg_norm)
        print("Number of iters:", lm_opt.nfev)
        print("Translation scale:", lm_opt.x[0])

        T = np.copy(T)
        T[:3, 3] = lm_opt.x[0] * T[:3, 3]

        return T, avg_norm

    else:
        return None


def average_transforms(T):
    R_quat = []
    t = []

    for Ti in T:
        R_quat.append(Rotation.from_matrix(Ti[:3, :3]).as_quat())
        t.append(Ti[:3, 3])

    R_quat = np.mean(np.array(R_quat), axis=0)
    t = np.mean(np.array(t), axis=0)

    T_final = np.zeros((4, 4))
    T_final[:3, :3] = Rotation.from_quat(R_quat).as_matrix()
    T_final[:3, 3] = t
    T_final[3, 3] = 1

    return T_final


"""
Utils
"""


def to_norm_image_coord(loc_kp, intrinsics):
    """
    :param loc_kp: N x 2
    :param intrinsics: 3 x 3
    """
    return (np.linalg.inv(intrinsics) @ to_homogeneous(loc_kp).transpose()).transpose()


def project2image(scene_points, intrinsics):
    """
    :param scene_points: N x 3
    :param intrinsics: 3 x 3
    """
    return to_cartesian((intrinsics @ scene_points.transpose()).transpose())


def transform2local(loc_kp, norm_loc_kp, depth):
    """
    :param loc_kp: N x 2
    :param norm_loc_kp: N x 2
    :param depth: H x W
    """
    loc_kp_depth = interpolate(depth, loc_kp)

    local_loc_kp = norm_loc_kp * np.expand_dims(loc_kp_depth.reshape(-1), axis=-1)

    return local_loc_kp


def interpolate(grid_values, points):
    grid_x, grid_y = np.meshgrid(np.arange(grid_values.shape[1]), np.arange(grid_values.shape[0]))
    grid = np.concatenate([np.expand_dims(grid_x, -1),
                           np.expand_dims(grid_y, -1)], axis=-1)

    return griddata(grid.reshape(-1, 2), grid_values.reshape(-1), points, method='nearest', fill_value=0)


def to_homogeneous(t):
    return np.concatenate((t, np.ones((len(t), 1))), axis=-1)


def to_cartesian(t):
    return t[:, :-1] / np.expand_dims(t[:, -1], -1)


def vec_to_cross(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def compose_ess_mat(T):
    R, t = T[:3, :3], T[:3, 3]

    cp_t = vec_to_cross(t)
    E = cp_t @ R

    return E


def loss_fun_tr(m, T, local_kp1, loc_kp2, K2):
    T = np.copy(T)
    T[:3, 3] = m * T[:3, 3]

    t_local_kp1 = to_cartesian((T @ to_homogeneous(local_kp1).transpose()).transpose())
    p_local_kp1 = project2image(t_local_kp1, K2)

    norm_diff = np.linalg.norm(p_local_kp1 - loc_kp2, axis=-1).reshape(-1)

    return norm_diff


def loss_fun_pose(x, kp1, kp2, iK1, iK2):
    E = decode_essmat(x)
    F = iK2.T @ E @ iK1
    errs, errs_rev = compute_epipolar_errors_opt(F, kp1, kp2)
    return np.concatenate([errs, errs_rev], axis=0)


def loss_fun_pose_jac(x, kp1, kp2, iK1, iK2):
    E, dE = decode_essmat(x, True)
    F = iK2.T @ E @ iK1
    dF = np.matmul(iK2.T.reshape(1, 3, 3), np.matmul(dE, iK1.reshape(1, 3, 3)))
    _, _, derrs, derrs_rev = compute_epipolar_errors_opt(F, kp1, kp2, dF)
    dres = np.concatenate([derrs.T, derrs_rev.T], axis=0)
    return dres


def compute_epipolar_errors_opt(F, kp1, kp2, dF=None):
    n = kp1.shape[0]
    kp1h = np.concatenate([kp1, np.ones((n, 1))], axis=1)
    kp2h = np.concatenate([kp2, np.ones((n, 1))], axis=1)
    luh = kp1h @ F.T
    lambdas = np.linalg.norm(luh[:, 0:2], axis=1).reshape(n, 1)
    lh = luh / lambdas
    errs = np.matmul(lh.reshape(n, 1, 3), kp2h.reshape(n, 3, 1)).reshape(-1)

    luh_rev = kp2h @ F
    lambdas_rev = np.linalg.norm(luh_rev[:, 0:2], axis=1).reshape(n, 1)
    lh_rev = luh_rev / lambdas_rev
    errs_rev = np.matmul(lh_rev.reshape(n, 1, 3), kp1h.reshape(n, 3, 1)).reshape(-1)
    if dF is None:
        return errs, errs_rev

    dFt = dF.transpose(0, 2, 1)
    derrs = diff_ptl_errs(kp1h, kp2h, luh, lambdas, dFt)
    derrs_rev = diff_ptl_errs(kp2h, kp1h, luh_rev, lambdas_rev, dF)

    return errs, errs_rev, derrs, derrs_rev


def diff_ptl_errs(kp1h, kp2h, luh, lambdas, dFt, F=None, lh=None):
    n = kp1h.shape[0]
    dluh = np.matmul(kp1h.reshape(1, n, 3), dFt)  # 5 x n x 3
    dlnh = (1.0 / lambdas).reshape(n, 1, 1) * np.eye(3).reshape(1, 3, 3)  # n, 3, 3
    lzuh = np.concatenate([luh[:, 0:2], np.zeros((n, 1))], axis=1)
    dlnh -= (lambdas ** (-3)).reshape(n, 1, 1) * np.matmul(luh.reshape(n, 3, 1), lzuh.reshape(n, 1, 3))
    dlh = np.matmul(dlnh.reshape(1, n, 3, 3), dluh.reshape(5, n, 3, 1)).reshape(5, n, 3)
    derrs = np.matmul(dlh.reshape(5, n, 1, 3), kp2h.reshape(1, n, 3, 1)).reshape(5, n)
    if F is None:
        return derrs
    derr_pt = diff_ptl_errs_pts(kp1h, kp2h, lh, dlh, dlnh, F, dFt, lambdas, luh, lzuh, dluh)
    return derrs, derr_pt


def diff_ptl_errs_pts(kp1h, kp2h, lh, dlh, dlnh, F, dFt, lambdas, luh, lzuh, dluh):
    n = kp1h.shape[0]

    derr2 = lh.transpose()[0:2, :]  # 3 x n
    d2err2 = dlh.reshape(5, n, 3).transpose(0, 2, 1)[:, 0:2, :]  # 5 x 3 x n

    derr = np.matmul(kp2h.reshape(n, 1, 3), np.matmul(dlnh.reshape(n, 3, 3),
                                                      F.reshape(1, 3, 3))).reshape(n, 3)[:, 0:2].T
    # d2err = np.matmul(kp2h.reshape(1, n, 1, 3),
    #                   np.matmul(dlnh.reshape(1, n, 3, 3),
    #                   dFt.reshape(5, 1, 3, 3).transpose(0, 1, 3, 2))).reshape(5, n, 3)[:, :, 0:2].transpose(0, 2, 1)
    luh_dx = F.transpose().reshape(1, 3, 3)
    dlambda = np.matmul(luh_dx, lzuh.reshape(n, 3, 1))  # n x 3 x 1
    d = np.eye(3).reshape(1, 1, 3, 3) * dlambda.reshape(n, 3, 1, 1) * (
        -lambdas.reshape(n, 1, 1, 1) ** (-3))  # n x 3 x 3 x 3
    d += 3 * lambdas.reshape(n, 1, 1, 1) ** (-5) * np.matmul(luh.reshape(n, 1, 3, 1),
                                                             lzuh.reshape(n, 1, 1, 3)) * dlambda.reshape(n, 3, 1, 1)

    d += -(lambdas ** (-3)).reshape(n, 1, 1, 1) * np.matmul(luh_dx.reshape(1, 3, 3, 1), lzuh.reshape(n, 1, 1, 3))
    F0 = np.copy(F)
    F0[2, :] = 0
    luh0_dx = F0.transpose().reshape(1, 3, 1, 3)
    d += -(lambdas ** (-3)).reshape(n, 1, 1, 1) * np.matmul(luh.reshape(n, 1, 3, 1), luh0_dx)

    # dluh: 5 x n x 3
    dmult = np.matmul(d.reshape(1, n, 3, 3, 3), dluh.reshape(5, n, 1, 3, 1)).reshape(5, n, 3, 3, 1)
    dmult += np.matmul(dlnh.reshape(1, n, 1, 3, 3), dFt.reshape(5, 1, 3, 3, 1)).reshape(5, n, 3, 3, 1)  # old

    d2err = np.matmul(kp2h.reshape(1, n, 1, 1, 3), dmult).reshape(5, n, 3)[:, :, 0:2].transpose(0, 2, 1)

    return derr, d2err, derr2, d2err2


def encode_essmat(T):
    R, t = T[:3, :3], T[:3, 3]

    R_param, jac = cv.Rodrigues(R)

    vec = np.asarray([0, 0, 1])
    t_param = rotate_a_b_axis_angle(vec, t)

    x = np.concatenate([R_param.reshape(-1), t_param[0:2]], axis=0)

    return x


def decode_essmat(x, is_jac=False):
    # assuming jac is 3 x 9
    R, jac = cv.Rodrigues(x[0:3])
    x2 = np.zeros(3)
    x2[0:2] = x[3:5]
    R2, jac2 = cv.Rodrigues(x2)
    t = R2[:, 2]
    cp_t = vec_to_cross(t)
    E = cp_t @ R
    if not is_jac:
        return E

    jac_big = np.zeros((5, 3, 3))
    jac_big[0:3, :, :] = np.matmul(cp_t.reshape(1, 3, 3), jac.reshape(3, 3, 3))

    dcpt1 = np.zeros((3, 3))
    dcpt1[1, 2] = -1
    dcpt1[2, 1] = 1
    jac_big[3:5, :, :] = (dcpt1 @ R).reshape(1, 3, 3) * jac2[0:2, 2].reshape(2, 1, 1)

    dcpt2 = np.zeros((3, 3))
    dcpt2[0, 2] = 1
    dcpt2[2, 0] = -1
    jac_big[3:5, :, :] += (dcpt2 @ R).reshape(1, 3, 3) * jac2[0:2, 5].reshape(2, 1, 1)

    dcpt3 = np.zeros((3, 3))
    dcpt3[0, 1] = -1
    dcpt3[1, 0] = 1
    jac_big[3:5, :, :] += (dcpt3 @ R).reshape(1, 3, 3) * jac2[0:2, 8].reshape(2, 1, 1)

    return E, jac_big


def rotate_a_b_axis_angle(a, b):
    a = a / np.clip(np.linalg.norm(a), a_min=1e-16, a_max=None)
    b = b / np.clip(np.linalg.norm(b), a_min=1e-16, a_max=None)
    rot_axis = np.cross(a, b)
    #   find a proj onto b
    a_proj = b * (a.dot(b))
    a_ort = a - a_proj
    #   find angle between a and b in [0, np.pi)
    theta = np.arctan2(np.linalg.norm(a_ort), np.linalg.norm(a_proj))
    if a.dot(b) < 0:
        theta = np.pi - theta

    aa = rot_axis / np.clip(np.linalg.norm(rot_axis), a_min=1e-16, a_max=None) * theta
    return aa


# def loss_fun_tr_batch(m, T, local_kp1, loc_kp2, K2):
#     T = np.copy(T)
#     T[:3, 3] = m * T[:3, 3]
#
#     norm_diff_batch = []
#
#     for local_kp1i, loc_kp2i, K2i in zip(local_kp1, loc_kp2, K2):
#         t_local_kp1i = to_cartesian((T @ to_homogeneous(local_kp1i).transpose()).transpose())
#         p_local_kp1i = project2image(t_local_kp1i, K2i)
#
#         norm_diff = np.linalg.norm(p_local_kp1i - loc_kp2i, axis=-1).reshape(-1)
#
#         norm_diff_batch.append(norm_diff)
#
#     return np.concatenate(norm_diff_batch, axis=0)


# def loss_fun_pose_batch(x, kp1, kp2, iK1, iK2):
#     errs_batch = []
#
#     for kp1i, kp2i, ik1i, ik2i in zip(kp1, kp2, iK1, iK2):
#         errs_batch.append(loss_fun_pose(x, kp1i, kp2i, ik1i, ik2i))
#
#     return np.concatenate(errs_batch, axis=0)


# def loss_fun_pose_jac_batch(x, kp1, kp2, iK1, iK2):
#     dres_batch = []
#
#     for kp1i, kp2i, ik1i, ik2i in zip(kp1, kp2, iK1, iK2):
#         dres_batch.append(loss_fun_pose_jac(x, kp1i, kp2i, ik1i, ik2i))
#
#     return np.concatenate(dres_batch, axis=0)
