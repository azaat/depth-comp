import cv2 as cv
import numpy as np
import open3d
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

from utils.calib_utils import project2image


def plot_figures(figures, nrows=1, ncols=1, size=(18, 18)):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axes_list = plt.subplots(ncols=ncols, nrows=nrows, figsize=size)
    for ind, title in zip(range(len(figures)), figures):
        if nrows * ncols != 1:
            axes_list.ravel()[ind].imshow(figures[title], cmap='gray')
            axes_list.ravel()[ind].set_title(title)
            axes_list.ravel()[ind].set_axis_off()
        else:
            axes_list.imshow(figures[title], cmap='gray')
            axes_list.set_title(title)
            axes_list.set_axis_off()

    plt.tight_layout()  # optional


def plot_projected_keypoints(image, local_scene_points, undist_intrinsics, key, pattern_size, fig_size=(9, 9)):
    proj_image_points = project2image(local_scene_points, undist_intrinsics)
    corners_image = draw_chessboard_corners(image, np.expand_dims(proj_image_points.astype(np.float32), 1), pattern_size)

    plt.figure(figsize=fig_size)
    plt.title(key)
    plt.imshow(corners_image, cmap='gray')


def plot_projected_pcd(image, local_scene_points, undist_intrinsics, key, fig_size=(18, 18)):
    h, w = image.shape[:2]
    pcd_image = np.zeros(image.shape)

    d = np.linalg.norm(local_scene_points, axis=-1)

    proj_pcd = project2image(local_scene_points, undist_intrinsics)
    proj_pcd = np.round(proj_pcd).astype(np.int)[:, [0, 1]]

    proj_mask = (proj_pcd[:, 0] >= 0) & (proj_pcd[:, 0] < w) & (proj_pcd[:, 1] >= 0) & (proj_pcd[:, 1] < h)

    proj_pcd = proj_pcd[proj_mask, :]
    d = d[proj_mask]

    pcd_image[proj_pcd[:, 1], proj_pcd[:, 0], 0] = 1 / d
    pcd_image[:, :, 0] = convolve2d(pcd_image[:, :, 0], np.ones((3, 3)), mode='same')

    plt.figure(figsize=fig_size)
    plt.title(key)
    plt.imshow(np.clip(pcd_image + image / 255, 0, 1))


def plot_epipolar_lines(image1, image2, loc_kp1, loc_kp2, key1, key2, F, pattern_size):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    ep_lines1 = cv.computeCorrespondEpilines(loc_kp2, 2, F).reshape(-1, 3)
    ep_lines2 = cv.computeCorrespondEpilines(loc_kp1, 1, F).reshape(-1, 3)

    ep_lines_image1 = draw_ep_lines(np.copy(image1), ep_lines1, loc_kp1)
    det_image2 = draw_chessboard_corners(np.copy(image2), loc_kp2, pattern_size)

    ep_lines_image2 = draw_ep_lines(np.copy(image2), ep_lines2, loc_kp2)
    det_image1 = draw_chessboard_corners(np.copy(image1), loc_kp1, pattern_size)

    axes[0][0].imshow(ep_lines_image1, cmap='gray')
    axes[0][0].set_title(key1)

    axes[0][1].imshow(det_image2, cmap='gray')
    axes[0][1].set_title(key2)

    axes[1][0].imshow(det_image1, cmap='gray')
    axes[1][0].set_title(key1)

    axes[1][1].imshow(ep_lines_image2, cmap='gray')
    axes[1][1].set_title(key2)


"""
Support utils
"""


def draw_chessboard_corners(image, loc_kp, pattern_size):
    det_img = np.copy(image)
    det_img = cv.drawChessboardCorners(det_img, pattern_size, loc_kp, True)

    return det_img


def draw_ep_lines(image, ep_line, loc_kp):
    c = image.shape[1]

    for l, lk in zip(ep_line, loc_kp):
        color = tuple(np.random.randint(0, 255, 3).tolist())

        x0, y0 = map(int, [0, -l[2] / l[1]])
        x1, y1 = map(int, [c, -(l[2] + l[0] * c) / l[1]])

        image = cv.line(image, (x0, y0), (x1, y1), color, 1, lineType=cv.LINE_AA)
        image = cv.circle(image, tuple((int(lk[0][0]), int(lk[0][1]))), 5, color, -1)

    return image


def normalize_image(image):
    if len(image.shape) == 3:
        return image / image.reshape(-1, 3).max(axis=0).reshape(1, 1, 3)

    elif len(image.shape) == 2:
        return image / image.max()

    else:
        raise NotImplementedError


def to_open3d(pcd):
    o3d_pcd = open3d.geometry.PointCloud()
    o3d_pcd.points = open3d.utility.Vector3dVector(pcd)

    return o3d_pcd
