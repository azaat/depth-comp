import cv2 as cv2
import numpy as np
import scipy
import open3d as o3d

azure_calib = './camera-depth-calibration/calib_output/'

# Point cloud from depth (by Konstantin)
def pointcloudify_depth(depth, intrinsics, dist_coeff, undistort=True):
    shape = depth.shape[::-1]
    
    if undistort:
        undist_intrinsics, _ = cv2.getOptimalNewCameraMatrix(intrinsics, dist_coeff, shape, 1, shape)
        inv_undist_intrinsics = np.linalg.inv(undist_intrinsics)

    else:
        inv_undist_intrinsics = np.linalg.inv(intrinsics)

    if undistort:
        # undist_depthi = cv2.undistort(depthi, intrinsics, dist_coeff, None, undist_intrinsics)
        map_x, map_y = cv2.initUndistortRectifyMap(intrinsics, dist_coeff, None
                                                  , undist_intrinsics, shape, cv2.CV_32FC1)
        undist_depth = cv2.remap(depth, map_x, map_y, cv2.INTER_NEAREST)

    # Generate x,y grid for H x W image
    grid_x, grid_y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    grid = np.concatenate([np.expand_dims(grid_x, -1),
                           np.expand_dims(grid_y, -1)], axis=-1)

    grid = np.concatenate([grid, np.ones((shape[1], shape[0], 1))], axis=-1)

    # To normalized image coordinates
    local_grid = inv_undist_intrinsics @ grid.reshape(-1, 3).transpose()  # 3 x H * W

    # Raise by undistorted depth value from image plane to local camera space
    if undistort:
        local_grid = local_grid.transpose() * np.expand_dims(undist_depth.reshape(-1), axis=-1)

    else:
        local_grid = local_grid.transpose() * np.expand_dims(depth.reshape(-1), axis=-1)
        
    return local_grid.astype(np.float32)


def project_pcd_to_depth(pcd, undist_intrinsics, img_size): 
    I = np.zeros(img_size, np.float32)
    h, w = img_size
#     for P in pcd.points:
#         d = np.linalg.norm(P)
#         P = P / P[2]
#         p = undist_intrinsics @ P
#         x, y = int(np.round(p[0])), int(np.round(p[1]))
#         if x >= 0 and x < w and y >= 0 and y < h:
#             I[y, x] = d
            
#     return I
    points = np.asarray(pcd.points)
    d = np.linalg.norm(points, axis=1)
    # print(d)
#     print(points.shape)
    normalized_points = points / np.expand_dims(points[:, 2], axis=1)
    proj_pcd = np.round(undist_intrinsics @ normalized_points.T).astype(np.int)[:2].T
    proj_mask = (proj_pcd[:, 0] >= 0) & (proj_pcd[:, 0] < w) & (proj_pcd[:, 1] >= 0) & (proj_pcd[:, 1] < h)
    proj_pcd = proj_pcd[proj_mask, :]
    d = d[proj_mask]
    pcd_image = np.zeros((h, w))
    pcd_image[proj_pcd[:, 1], proj_pcd[:, 0]] = d
    return pcd_image


def smooth_depth(depth):
    MAX_DEPTH_VAL = 1e5
    KERNEL_SIZE = 3
    depth[depth == 0] = MAX_DEPTH_VAL
    smoothed_depth = scipy.ndimage.minimum_filter(depth, KERNEL_SIZE)
    smoothed_depth[smoothed_depth == MAX_DEPTH_VAL] = 0
    return smoothed_depth

def align_rgb_depth(rgb, depth, roi, dist_mtx, dist_coef):
    T = np.load(f'{azure_calib}azure2s20_standard_extrinsics.npy')
    _align_rgb_depth(rgb, depth, roi, T, dist_mtx, dist_coef)


def _align_rgb_depth(undist_rgb, depth, roi, T, undist, dist_mtx, dist_coef):
    # Undistort rgb image
    rgb_cnf = np.load(f'{azure_calib}s20_wide_intrinsics.npy', allow_pickle=True).item()
#     undist_rgb = cv2.undistort(rgb, config_dict['rgb']['dist_mtx'], config_dict['rgb']['dist_coef'],
#                               None, config_dict['rgb']['undist_mtx'])
    
#     undist_rgb = cv2.undistort(rgb, rgb_cnf['intrinsics'], rgb_cnf['dist_coeff'],
#                               None, rgb_cnf['undist_intrinsics'])

    # Create point cloud from depth
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloudify_depth(depth, dist_mtx,
                                                dist_coef))

    # Align point cloud with depth reference frame
    pcd.transform(T)

#     pcd.transform(T)
#     h, w = 1080, 1920
#     local_scene_points = to_cartesian((T @ to_homogeneous(points).transpose()).transpose())
#     d = np.linalg.norm(local_scene_points, axis=-1)
#     proj_pcd = project2image(local_scene_points, rgb_cnf['undist_intrinsics'])
#     proj_pcd = np.round(proj_pcd).astype(np.int)[:, [0, 1]]
#     proj_mask = (proj_pcd[:, 0] >= 0) & (proj_pcd[:, 0] < w) & (proj_pcd[:, 1] >= 0) & (proj_pcd[:, 1] < h)

#     proj_pcd = proj_pcd[proj_mask, :]
#     d = d[proj_mask]

#     pcd_image = np.zeros((1080, 1920, 3))
#     pcd_image[proj_pcd[:, 1], proj_pcd[:, 0], 0] = d
#     pcd_image[:, :, 0] = convolve2d(pcd_image[:, :, 0], np.ones((3, 3)), mode='same')
    
#     d = np.linalg.norm(np.asarray(pcd.points), axis=-1)
#     proj_pcd = project2image(np.asarray(pcd.points), config_dict['rgb']['undist_mtx'])
#     aligned_depth = np.round(proj_pcd).astype(np.int)[:, [0, 1]]
#     print(aligned_depth)
#     pcd_image[proj_pcd[:, 1], proj_pcd[:, 0], 0] = d
    
    # Project aligned point cloud to rgb
     
    aligned_depth = project_pcd_to_depth(pcd, undist, undist_rgb.shape[:2])
    
#     smoothed_aligned_depth = smooth_depth(aligned_depth)
    smoothed_aligned_depth = aligned_depth
    x, y, w, h = roi

    # depth_res = np.zeros((360, 640))
    # depth_res[y:y+h, x:x+w] = smoothed_aligned_depth[y:y+h, x:x+w]
    return smoothed_aligned_depth

def project2image(scene_points, intrinsics):
    """
    :param scene_points: N x 3
    :param intrinsics: 3 x 3
    """
    return to_cartesian((intrinsics @ scene_points.transpose()).transpose())

def to_cartesian(t):
    return t[:, :-1] / np.expand_dims(t[:, -1], -1)

def to_homogeneous(t):
    return np.concatenate((t, np.ones((len(t), 1))), axis=-1)