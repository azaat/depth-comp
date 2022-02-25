import numpy as np


def filter_orientation(detections, pattern_size, anchor_index=0):
    f_detections = {}

    keys = list(detections.keys())
    anchor_key = keys[anchor_index]

    f_detections[anchor_key] = detections[anchor_key]

    anchor_loc_kp = np.squeeze(detections[anchor_key][1], axis=1)
    anchor_is_bottom = (anchor_loc_kp[pattern_size[0]] - anchor_loc_kp[0])[1] > 0
    anchor_is_right = (anchor_loc_kp[1] - anchor_loc_kp[0])[0] > 0

    keys = [key for i, key in enumerate(keys) if i != anchor_index]

    for key in keys:
        scene_points, loc_kp = detections[key]
        loc_kp = np.squeeze(loc_kp, axis=1)

        is_bottom = (loc_kp[pattern_size[0]] - loc_kp[0])[1] > 0
        is_right = (loc_kp[1] - loc_kp[0])[0] > 0

        if anchor_is_bottom != is_bottom:
            scene_points = np.flip(scene_points, axis=0)

        if anchor_is_right != is_right:
            scene_points = np.flip(scene_points, axis=1)

        f_detections[key] = (scene_points, loc_kp)

    return f_detections


def prepare_calib_input(images, detections, keys, ext=''):
    scene_points = []
    loc_kp = []

    keys = [key + ext for key in keys]

    for key in keys:
        det = detections[key]
        scene_points.append(det[0])
        loc_kp.append(det[1])

    shape = images[keys[0]].shape[::-1][1:]

    return scene_points, loc_kp, shape

