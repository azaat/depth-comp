import numpy as np
import cv2
import matplotlib.pyplot as plt


def depth_map_SGBM(imgL_color, imgR_color, min_disp, max_disp, block_size=1):
    imgL_ = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2GRAY)
    imgR_ = cv2.cvtColor(imgR_color, cv2.COLOR_BGR2GRAY)

    # Maximum disparity minus minimum disparity. The value is always greater than zero.
    # In the current implementation, this parameter must be divisible by 16.
    num_disp = max_disp - min_disp
    # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
    # Normally, a value within the 5-15 range is good enough
    uniquenessRatio = 5
    # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
    # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
    speckleWindowSize = 30
    # Maximum disparity variation within each connected component.
    # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
    # Normally, 1 or 2 is good enough.
    speckleRange = 2
    disp12MaxDiff = 1

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=8 * 3 * block_size * block_size,
        P2=32 * 3 * block_size * block_size,
        mode=cv2.STEREO_SGBM_MODE_SGBM
    )
    displ = left_matcher.compute(imgL_, imgR_)
    return displ


# Visualize epilines
# Adapted from: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
def drawlines(img1src, img2src, lines, pts1src, pts2src):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1src.shape
    img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
    img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
    return img1color, img2color


def eq_hist(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    plt.imshow(img_output)
    plt.show()
    return img


def load_calibration_params(input_shape, calib_scaling):
    intrinsics_l = dict(np.load('intrinsic_data_left.npz'))
    intrinsics_r = dict(np.load('intrinsic_data_right.npz'))
    extrinsics = dict(np.load('extrinsic_data.npz'))

    # scale calibration values according to the parameter (
    # e.g. scaling = 1/3 for downscaled 3 times)
    extrinsics['proj_mat_l'] = extrinsics['proj_mat_l']*calib_scaling
    extrinsics['new_mtxL'] = extrinsics['new_mtxL']*calib_scaling

    extrinsics['proj_mat_l'][2, 2] = 1
    extrinsics['new_mtxL'][2, 2] = 1

    extrinsics['proj_mat_r'] = extrinsics['proj_mat_r']*calib_scaling
    extrinsics['new_mtxR'] = extrinsics['new_mtxR']*calib_scaling

    extrinsics['proj_mat_r'][2, 2] = 1
    extrinsics['new_mtxR'][2, 2] = 1

    extrinsics['Q'] = extrinsics['Q']*calib_scaling
    extrinsics['Q'][0, 0] = 1
    extrinsics['Q'][1, 1] = 1
    extrinsics['Q'][3, 2] = extrinsics['Q'][3, 2]/calib_scaling

    Left_Stereo_Map = cv2.initUndistortRectifyMap(extrinsics['new_mtxL'],
                                                  extrinsics['distL'],
                                                  extrinsics['rect_l'],
                                                  extrinsics['proj_mat_l'],
                                                  input_shape, cv2.CV_16SC2)
    Right_Stereo_Map = cv2.initUndistortRectifyMap(extrinsics['new_mtxR'],
                                                   extrinsics['distR'],
                                                   extrinsics['rect_r'],
                                                   extrinsics['proj_mat_r'],
                                                   input_shape, cv2.CV_16SC2)
    return Left_Stereo_Map, Right_Stereo_Map, extrinsics, \
        intrinsics_l, intrinsics_r


def rect(imgL, imgR, roi, stereo_maps, scaling_l=1, scaling_r=1, display=False):
    Left_Stereo_Map, Right_Stereo_Map = stereo_maps
    x, y, w, h = roi
    imgL = cv2.resize(imgL, (0, 0), fx=scaling_r, fy=scaling_l)

    tmp = np.zeros((w, h, 3))
    tmp = imgL[y:y+h, x:x+w, :]
    imgL = tmp

    imgL_rectified = cv2.remap(imgL, Left_Stereo_Map[0], Left_Stereo_Map[1],
                               cv2.INTER_LINEAR)
    imgR = cv2.resize(imgR, (0, 0), fx=scaling_r, fy=scaling_r)

    imgR_rectified = cv2.remap(imgR, Right_Stereo_Map[0], Right_Stereo_Map[1],
                               cv2.INTER_LINEAR)
    if display:
        fig, axes = plt.subplots(1, 2, figsize=(50, 20))
        axes[0].imshow(imgL_rectified, cmap="gray")
        axes[1].imshow(imgR_rectified, cmap="gray")

        plt.suptitle("Rectified images")
        plt.show()
    return (imgL_rectified, imgR_rectified)

def extract_depth_arcore(x):
    depthConfidence = (x >> 13) & 0x7
    if (depthConfidence > 6):
        return 0 
    return x & 0x1FFF
