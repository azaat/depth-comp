import os
import re
import open3d
import matplotlib.pyplot as plt
import numpy as np


def get_matching_pairs(folder_path1, folder_path2, file_filter='\d{5}.*\.(png|jpg|jpeg)$'):
    file_names1, file_names2 = sorted(filter_files(os.listdir(folder_path1), file_filter), key=str.lower), \
                               sorted(filter_files(os.listdir(folder_path2), file_filter), key=str.lower)

    timestamps1, timestamps2 = np.array([int(fn.split('.')[0]) for fn in file_names1]), \
                               np.array([int(fn.split('.')[0]) for fn in file_names2])

    diff = np.abs(timestamps1.reshape(-1, 1) - timestamps2.reshape(1, -1))
    nn_indices1 = np.argmin(diff, axis=-1)
    nn_indices2 = np.argmin(diff, axis=-2)

    mnn_mask = nn_indices2[nn_indices1] == np.arange(len(nn_indices1))

    timestamps1 = timestamps1[mnn_mask]
    timestamps2 = timestamps2[nn_indices1][mnn_mask]

    return timestamps1.astype(np.str), timestamps2.astype(np.str)


"""
Functions for reading the data
"""


def get_images(folder_path, start=None, period=None, file_name_list=None, is_inverted=False):
    """
    :param folder_path: path to the folder with images
    :param period: the rate of considering images; used to reduce the final number of images
    :param file_name_list: the list of file names to consider; used to provide manually selected images
    """
    return get_data(load_image, folder_path, start, period, file_name_list)


def get_pointclouds(folder_path, start=None, period=None, file_name_list=None):
    """
    :param folder_path: path to the folder with pointclouds
    :param start: pointcloud to start from
    :param period: the rate of considering pointclouds
    :param file_name_list: the list of file names to consider
    """
    return get_data(load_pcd, folder_path, start, period, file_name_list)


def get_depths(folder_path, start=None, period=None, file_name_list=None, is_inverted=False):
    """
    :param folder_path: path to the folder with depths
    :param start: depth to start from
    :param period: the rate of considering depths
    :param file_name_list: the list of file names to consider
    """
    return get_data(load_depth, folder_path, start, period, file_name_list)


def get_data(data_loader, folder_path, start, period, file_name_list):
    """
    :param data_loader: function for loading data
    :param folder_path: folder to load data from
    :param start: the position from which to start reading data
    :param period: period to consider each i-th data sample
    :param file_name_list: list of files to load
    """
    file_names = sorted(os.listdir(folder_path), key=lambda s: s.lower())

    data = {}

    if file_name_list is None:
        for i, fname in enumerate(file_names):
            if start is not None and i < start:
                continue

            if period is not None and i % period != 0:
                continue

            fpath = os.path.join(folder_path, fname)

            datai = data_loader(fpath)

            if datai is not None:
                data[fname] = datai

    else:
        for i, fname in enumerate(file_name_list):
            if period is not None and i % period != 0:
                continue

            fpath = os.path.join(folder_path, fname)

            datai = data_loader(fpath)

            if datai is not None:
                data[fname] = datai

    return data


"""
Functions for saving the data
"""


def save_intrinsics_calib(calib_name, intrinsics, dist_coeff, undist_intrinsics):
    calib_fpath = 'multiview-camera-depth-calibration/calib_output/%s_intrinsics' % calib_name

    calib = {'intrinsics': intrinsics,
             'dist_coeff': dist_coeff,
             'undist_intrinsics': undist_intrinsics}

    np.save(calib_fpath, calib)

    print('Saved calibration results as %s.npy' % calib_fpath)


def save_extrinsics_calib(calib_name, T):
    calib_fpath = 'calib_output/%s_extrinsics' % calib_name
    calib = {'T': T}

    np.save(calib_fpath, calib)
    print('Saved calibration results as %s.npy' % calib_fpath)


"""
Support utils
"""


def load_image(file_path):
    if file_path.lower().endswith('jpg') or \
            file_path.lower().endswith('jpeg') or \
            file_path.lower().endswith('png'):

        img = plt.imread(file_path)

        if len(img.shape) == 3 and img.dtype == np.float32 and img.max() <= 1:
            img = (img * 255).astype(np.uint8)

        # Infra-red image
        if len(img.shape) == 2 and img.dtype == np.float32 and img.max() <= 1:
            img = (img / img.max() * 255).astype(np.uint8)

    else:
        return None

    return img


def load_pcd(file_path):
    if file_path.lower().endswith('pcd'):
        pcd = np.asarray(open3d.io.read_point_cloud(file_path).points).astype(np.float32)

    else:
        return None

    return pcd


def load_depth(file_path):
    if file_path.lower().endswith('npy'):
        pcd = np.load(file_path)

    else:
        return None

    return pcd


def filter_files(file_names, file_filter):
    return [fn for fn in file_names if re.match(file_filter, fn)]