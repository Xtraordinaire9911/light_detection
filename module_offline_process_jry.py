"""Support for offline batch process

The input offline batch test is:
    1. images of current collection and reference
    2. gps, attitude and heading records of current collection and reference

The steps of offline process
    1. generate metadata for all images
    2. iterate all images for:
        1). find best neighbor
        2). stitch and compare to find anomaly
G
The folder structure of collection is:
root folder:
    img:
        cam1:
        cam2:
        ...
    pixhawk:
        att_measures (file)
        gps_measures.csv (file)
        heading_measures (file)

Limitation:
    1. Require to indicate the compared image sets. Normally, the ref image to compare is chosen
    automatically according to localization and attitude. So we have to indicate the image set by
    camera serial.

"""
from typing import AnyStr
import pathlib
import os
import glob
import time
# import threading

import pandas as pd
import numpy as np
import time
import cv2

from runwayfod.util import func_preprocess as fpp
from runwayfod.util.class_ImgDiffDetector import AirportRunwayFFCmp
from copy import  deepcopy

IMG_EXT = ".npy"
FLAG_VIS = not True
FLAG_CROP_OBJ = True
FLAG_SAVE = True
READ_NEIGHBOR = not True  # False
SAVE_GOOGLE_DRIVE = False


def generate_folder_path(dir: AnyStr):
    """
    Generate the path information for following process.
    In the processing functions, the input variables are independent to the config. So they can be
    reused easily.

    Args:
        dir:

    Returns:
        a diction of paths

    """
    paths = {}
    userexp = pathlib.Path(os.path.expanduser("~"))
    paths["dir"] = userexp / dir
    paths["sensor_data"] = paths["dir"] / "pixhawk" / "gps_measures.csv"
    paths["image"] = paths["dir"] / "img"

    return paths


def load_img(file_path, mask=None):
    ext = os.path.splitext(file_path)[-1]
    if ext == '.npy':
        img = np.load(file_path).astype(np.uint8)
        # print('the npy img shape', img.ndim)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
    elif ext == '.jpg':
        img = cv2.imread(file_path)
    else:
        img = None
    img = cv2.bitwise_and(img, img, mask=mask)
    return img


def load_gps(file_path):
    """

    Args:
        file_path:

    Returns:

    """
    gps_df = pd.read_csv(file_path)
    gps = gps_df[["ts", "lat", "lon"]]

    return gps


def generate_metadata(img_ts, sensor_data):
    """Create initial metadata from sensor data

    Args:
        img_ts: a list or array contains the timestamps of images
        sensor_data: a dict contains several dataframe

    Returns:
        metadata: a dict contains several metadata corresponding to the sensor_data


    """
    ts = sensor_data["ts"].values
    lon = sensor_data["lon"].values
    lat = sensor_data["lat"].values

    new_lon = np.interp(img_ts, ts, lon)
    new_lat = np.interp(img_ts, ts, lat)

    return np.hstack([img_ts, new_lat, new_lon])


def calculate_gps_distance(curr_gps, cmp_gps, is_in_degree=True):
    """

    Args:
        curr_gps: (lon, lat)
        cmp_gps: (lon, lat)
        is_in_degree:

    Returns:

    """
    R = 6373000.0

    # if None in curr_gps:
    #     return None
    lat_curr, lon_curr = curr_gps
    lat_cmp, lon_cmp = cmp_gps
    if is_in_degree:
        lon_curr = np.radians(lon_curr)
        lat_curr = np.radians(lat_curr)
        lon_cmp = np.radians(lon_cmp)
        lat_cmp = np.radians(lat_cmp)

    dlon = lon_cmp - lon_curr
    dlat = lat_cmp - lat_curr

    a = np.square(np.sin(dlat / 2)) + np.multiply(np.cos(lat_curr),
                                                  np.multiply(np.cos(lat_cmp),
                                                              np.square(np.sin(dlon / 2))
                                                              )
                                                  )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c

    return np.ravel(distance)


def gen_img_to_preload(curr_gps, ref_gps, preload_radius, is_in_degree=True):
    """

    Args:
        is_in_degree:
        curr_gps:
        ref_gps:
        preload_radius:

    Returns:
        preload_img_list

    """
    dist_to_ref_gps = calculate_gps_distance((curr_gps['lon'], curr_gps['lat']),
                                             (ref_gps['lon'].values, ref_gps['lat'].values), is_in_degree)
    idx = dist_to_ref_gps < preload_radius

    return np.where(idx)


def thread_preload_ref(img_buffer, curr_gps, ref_gps, ref_img_dir, preload_radius, end_event):
    while not end_event.is_set():
        img_idx = gen_img_to_preload(curr_gps, ref_gps, preload_radius)
        img_to_load = ref_gps["ts"].values[img_idx]
        img_to_load = set(img_to_load)
        new_imgs = set(img_to_load) - set(img_buffer.keys())
        new_imgs = list(new_imgs)
        new_imgs.sort()
        timeout_imgs = set(img_buffer.keys()) - set(img_to_load)
        for element in timeout_imgs:
            img_buffer.pop(element)
        for element in new_imgs:
            img_buffer[element] = load_img(os.path.join(ref_img_dir, str(element) + IMG_EXT))


def adjust_delta_gps_cv(gps_curr, img_ref, gps_ref):
    """If the current image is classified to a certain pattern we need, this function is used to
    update the delta gps.

    img_ref is a set of images, find the one which has the highest probability to have the same
    pattern

    Args:
        gps_curr: the gps coordinates of current image
        img_ref: a set of images which are close to current image (within a threshold)
        gps_ref: the gps of the set of images

    Returns:
        delta_gps: gps_ref_selected - gps_curr
    """


def find_best_gps_neighbor(gps_curr, gps_ref, dist_threshold):
    """Get the best neighbor within a threshold

    Args:
        gps_curr: (latitude, longitude)
        gps_ref: array of n * 2 for n reference (latitude, longitude)
        dist_threshold:

    Returns:
        idx: the index of best reference in gps_ref, -1 if no good result
    """
    dist_to_ref_gps = calculate_gps_distance((gps_curr['lon'], gps_curr['lat']),
                                             (gps_ref['lon'].values, gps_ref['lat'].values))
    print('the mindistance and the threshold:', np.min(dist_to_ref_gps), dist_threshold)

    if np.min(dist_to_ref_gps) < dist_threshold:
        return gps_ref["ts"].values[np.argmin(dist_to_ref_gps)]
    else:
        return None


def save_intermediate(imgs, save_dir, curr_img_name, ref_img_name):
    """Save a set of images (obj_img, ref, diff, diff_bin, erosion, dilation)

    Args:
        imgs: images stored in a tuple the order is
        save_dir:
        curr_img_name:
        ref_img_name:

    Returns:

    """
    cv2.imwrite(os.path.join(save_dir, "{}_{}_1_curr_img.jpg".format(curr_img_name, ref_img_name)),
                imgs[0])
    cv2.imwrite(
        os.path.join(save_dir, "{}_{}_2_ref_img_transformed.jpg".format(curr_img_name, ref_img_name)),
        imgs[1])
    cv2.imwrite(os.path.join(save_dir, "{}_{}_3_img_diff.jpg".format(curr_img_name, ref_img_name)),
                cv2.absdiff(imgs[1], imgs[2]))

    cv2.imwrite(os.path.join(save_dir, "{}_{}_4_diff_bin.jpg".format(curr_img_name, ref_img_name)),
                imgs[3])
    cv2.imwrite(
        os.path.join(save_dir, "{}_{}_5_diff_erosion.jpg".format(curr_img_name, ref_img_name)),
        imgs[4])
    cv2.imwrite(os.path.join(save_dir, "{}_{}_6_img_dilation.jpg".format(curr_img_name, ref_img_name)),
                imgs[5])


def stitch_cmp(detector, tmp_img_dict, curr_img, id_obj, id_ref, cfg_sys):
    """stitch and compare the input 2 images with detector and config

    Args:
        detector:
        curr_img:
        ref_img:
        curr_name:
        ref_name:
        feature:
        cfg_sys:

    Returns:

    """
    curr_name = id_obj
    ref_name = id_ref
    feature = cfg_sys["Features"]
    if tmp_img_dict is None:
        detector.logger.info("Stitch is failed for {} and {}".format(curr_name, ref_name))
        return
    h, w = curr_img.shape[0: 2]
    tmp_res = detector.flip_flop_compare(tmp_img_dict["scr_p"], tmp_img_dict["dst_p"], tmp_img_dict["delta"],
                                         (h, w), color_space="gray",
                                         pre_kernel_size=int(feature["pre_kernel_size"]),
                                         flag_im_show=detector.flag_show_flip_flop_process,
                                         bi_thred=detector.bi_thred, max_val=255)
    pnts_rect_list, img_diff_overlap_bi, erosion, dilation = tmp_res
    if FLAG_CROP_OBJ and pnts_rect_list is not None:
        detector.batch_crop_detect_object(pnts_rect_list, curr_img, curr_name, ref_name)
    if FLAG_SAVE:
        tmp_imgs = (tmp_img_dict["dst_p"], tmp_img_dict["scr_p"],
                    cv2.absdiff(tmp_img_dict["dst_p"], tmp_img_dict["scr_p"]), img_diff_overlap_bi,
                    erosion, dilation,)
        tmp_dir = os.path.join(detector.output_path, "intermediate")
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        save_intermediate(tmp_imgs, tmp_dir, curr_name, ref_name)
    return tmp_img_dict


def set_delta_gps(obs_gps, gt_gps):
    """Update the GPS delta between ground truth and current detected value

    Args:
        curr_gps: current observed gps value (lat, lon)
        gt_gps: the ground truth of gps value (lat, lon)

    Returns:
        delta_gps: (delta_lat, delta_lon)
    """
    delta_lat = gt_gps[0] - obs_gps[0]
    delta_lon = gt_gps[1] - obs_gps[1]

    return delta_lat, delta_lon


def update_obs_gps(obs_gps, delta_gps):
    """Update observation GPS with delta GPS

    Args:
        obs_gps: observation GPS, np.ndarray n*2 (lat, lon)
        delta_gps: GPS delta between ground truth and observation, np.ndarray, 1* 2 (lat, lon)

    Returns:
        obs_gps_update: updated observation GPS
    """
    if obs_gps is None:
        return None

    obs_gps = np.asarray(obs_gps).reshape(-1, 2)

    obs_gps_update = obs_gps + delta_gps

    return obs_gps_update


def check_relative_pos_side_back(homomtx, img_dim):
    """Return the relative loc of 2 images with homomatrix and the dimension of source image
    The bottom right point of an image is applied by the transformation of homomatrix to measure
    the relative position between source and destination image.
    Args:
        homomtx: homomatrix between destination image and source image
        img_dim: dimension of source image (height, width)

    Returns:
        rel_pos: > 0, means source image is on the right of destination image
                 < 0, means source image is on the left of destination image
                 the absolute value of rel_pos measures the distortion os source image
    """
    h, w = img_dim
    pts2 = np.float32([[0, 0], [w, h]]).reshape(-1, 1, 2)
    pts_t = cv2.perspectiveTransform(pts2, homomtx)
    # print('the pts_t is \n', (pts_t[1]-pts2[1]).reshape(2,1))
    dw, dh = (pts_t[1] - pts_t[0]).reshape(2, 1)
    print('the corner after transforming:', pts_t)
    rel_pos_x = dw / w - 1
    rel_pos_y = dh / h - 1
    return rel_pos_x, rel_pos_y  # rel_shift.reshape(-1)


def find_dy_sign(detector, sys_config, ref_img_paths, nref=5):
    """
     dy_sign is solely determined by the reference 
     take a Homomatrix of 1st (ref) and 2ed (obj) photos from 
     reference, see the sign of dy. 
     if dy > 0: when dy>0 we need to do  n_ref = nref+1 
     if dy < 0: when dy<0 we need to do n_ref = nref+1
    """
    ref_img = load_img(ref_img_paths[nref])
    obj_img = load_img(ref_img_paths[nref + 1])
    homomat = detector.stitcher.stitch_homo((obj_img, ref_img), sys_config)
    _, dy = check_relative_pos_side_back(homomat, obj_img.shape[:2])
    return int(abs(dy) / dy)


def generate_neighbor_array(n_neighbors, n_ref_0, ref_total, dy_sign, num_comparison = 20):
    # a = np.array([-1, 1])
    # b = a
    # for i in range(2, n_neighbors + 1):
    #     b = np.append(b, a * i)
    # c = []
    # for j, ref in enumerate(b):
    #     n_ref = n_ref_0 + int(ref * dy_sign)
    #     if n_ref < ref_total:
    #         c.append(ref)
    # return np.array(c)
    start = max(- int(n_neighbors / 2) + n_ref_0, 0)
    end = min(int(n_neighbors / 2) + n_ref_0 + 1, ref_total - 1)
    return np.arange(start, end, 1)


def final_decision_dx_dy(sys_config, table, idx, dx, dy, n_ref, img_dict, dy_threshold_2):
    """
    decide if dx and dy are accepted and return the corresponding information
    """
    print('the final decision dx dy is', idx, n_ref, dx, dy, dy_threshold_2)
    dx_threshold_2 = float(sys_config['Neighbors']['dx_threshold_min'])  # 0.2
    dy_threshold_1 = float(sys_config['Neighbors']['dy_threshold_max'])  # 0.6
    if dx <= dx_threshold_2 and dy <= dy_threshold_1:
        table[idx, 6] = n_ref
        table[idx, 7] = dx
        table[idx, 8] = dy
        table[idx, 9] = dy_threshold_2
        img_dict['best_ref'] = n_ref
        return table, img_dit
    else:
        table[idx, 6] = n_ref
        table[idx, 7] = dx
        table[idx, 8] = dy
        table[idx, 9] = dy_threshold_2
        img_dict = None
        return table, img_dict


def decision_from_dx_dy(sys_config, tmp_dicts, idx, dx, dy, n_ref, homomat, dy_threshold_2):
    """
    with the final dx and dy list, decide reject or accept the neighbor
    """
    print('the final decision dx dy is', idx, n_ref, dx, dy, dy_threshold_2)
    tmp_dx = tmp_dicts['tmpdx']
    tmp_dy = tmp_dicts['tmpdy']
    tmp_n_ref = tmp_dicts['tmpnref']
    tmp_hmat = tmp_dicts['tmpmat']
    tmp_n_ref.append(n_ref)
    tmp_dy.append(abs(dy))
    tmp_dx.append(abs(dx))
    dx_threshold_2 = float(sys_config['Neighbors']['dx_threshold_min'])  # 0.2
    dy_threshold_1 = float(sys_config['Neighbors']['dy_threshold_max'])  # 0.6
    if dx <= dx_threshold_2 and dy <= dy_threshold_1:
        find_neighbor = True
        tmp_hmat.append(homomat)
        return find_neighbor, tmp_dicts
    else:
        find_neighbor = True
        homomat = None
        tmp_hmat.append(homomat)
        return find_neighbor, tmp_dicts


def stitch_failed(detector, sys_config, tmp_dicts, obj_img, idx, n_ref, ref_img_paths, dy_sign, mask, dx, dy, num_comparison = 20):
    """
    This function handel the situation when the stitch is completely failed,
    i.e., dx=dy=-1 or dx > dx_threshold_max
    """
    delta = float(sys_config['Neighbors']['tolerance'])  # tolerance for dy = -1.0 \pm delta
    dx_threshold_1 = float(sys_config['Neighbors']['dx_threshold_max'])  # 0.5
    dx_threshold_2 = float(sys_config['Neighbors']['dx_threshold_min'])  # 0.2
    dy_threshold_1 = float(sys_config['Neighbors']['dy_threshold_max'])  # 0.6
    dy_threshold_2 = float(sys_config['Neighbors']['dy_threshold_min'])  # 0.2
    n_neighbors = int(sys_config['Neighbors']['n_check'])

    tmp_dx = tmp_dicts['tmpdx']
    tmp_dy = tmp_dicts['tmpdy']
    tmp_n_ref = tmp_dicts['tmpnref']
    tmp_hmat = tmp_dicts['tmpmat']

    ref_total = len(ref_img_paths)
    neighbor_array = generate_neighbor_array(n_neighbors, n_ref, ref_total, dy_sign, num_comparison)

    print('enter stitch_failed function', idx, n_ref, dx, dy, dy_threshold_2)

    n_ref_0 = n_ref
    # for i in neighbor_array:
    for n_ref in np.arange(max(n_ref_0 - neighbor_array[0], 0), min(n_ref_0 + neighbor_array[-1] + 1, len(ref_img_paths)), 1):
        # n_ref = n_ref_0 + int(i * dy_sign)
        ref_img_path = ref_img_paths[n_ref]
        ref_img = load_img(ref_img_path, mask=mask)  # np.load(ref_img_path)
        homomat = detector.stitcher.stitch_homo((obj_img, ref_img), sys_config)
        if homomat is None:
            continue
        dx, dy = check_relative_pos_side_back(homomat, obj_img.shape[:2])
        print("the dx, and dy values for failed run 1st", idx, n_ref, dx, dy, dy_threshold_2)
        # if abs(dx) < dx_threshold_2 and abs(dy) < dy_threshold_2:
        #     print("the dx, and dy values final accepted", idx, n_ref, dx, dy, dy_threshold_2)
        #     tmp_n_ref.append(n_ref)
        #     tmp_dy.append(dy)
        #     tmp_dx.append(dx)
        #     tmp_hmat.append(homomat)
        #     find_neighbor = True
        #     return find_neighbor, tmp_dicts
        # elif abs(dx) < dx_threshold_2 and abs(dy) < dy_threshold_1:
        #     print("the dx, and dy values for failed run 2ed", idx, n_ref, dx, dy, dy_threshold_2)
            # dy_threshold_2 += 0.1  #
        tmp_n_ref.append(n_ref)
        tmp_dy.append(dy)
        tmp_dx.append(dx)
        tmp_hmat.append(homomat)
        # find_neighbor = False  # for fine tune
        find_neighbor = True
        # if i == neighbor_array[-1]:
        if n_ref == neighbor_array[-1]:
            print('Always dx=dy=-1!')
            homomat = None
        return find_neighbor, tmp_dicts
        # if i == neighbor_array[-1]:
        #     print('Always dx=dy=-1!')
        #     homomat = None
        #     tmp_n_ref.append(n_ref)
        #     tmp_dy.append(dy)
        #     tmp_dx.append(dx)
        #     tmp_hmat.append(homomat)
        #     find_neighbor = True
        #     return find_neighbor, tmp_dicts


def negative_dy(detector, sys_config, tmp_dicts, obj_img, idx, n_ref, ref_img_paths, dy_sign, mask, dx, dy):
    """
    handel the situation when dy> 0 in find the best neighbor function.
    """
    print('the dy is negative!!')
    delta = float(sys_config['Neighbors']['tolerance'])  # tolerance for dy = -1.0 \pm delta
    dx_threshold_1 = float(sys_config['Neighbors']['dx_threshold_max'])  # 0.5
    dx_threshold_2 = float(sys_config['Neighbors']['dx_threshold_min'])  # 0.2
    dy_threshold_1 = float(sys_config['Neighbors']['dy_threshold_max'])  # 0.6
    dy_threshold_2 = float(sys_config['Neighbors']['dy_threshold_min'])  # 0.2

    tmp_dx = tmp_dicts['tmpdx']
    tmp_dy = tmp_dicts['tmpdy']
    tmp_n_ref = tmp_dicts['tmpnref']
    tmp_hmat = tmp_dicts['tmpmat']

    ref_total = len(ref_img_paths)
    n_check = 3
    for i in range(n_check):  # it is enough to shift 2 or 3 next neighbors
        n_ref = n_ref - int(dy_sign)  # check the sign
        if n_ref >= ref_total:
            print('REACH THE LAST REFERENCE IMG!!!')
            tmp_n_ref.append(n_ref)
            tmp_dy.append(dy)
            tmp_dx.append(dx)
            homomat = None
            tmp_hmat.append(homomat)
            find_neighbor = True  # for fine tune
            return find_neighbor, tmp_dicts
        ref_img_path = ref_img_paths[n_ref]
        ref_img = load_img(ref_img_path, mask=mask)  # np.load(ref_img_path)
        homomat = detector.stitcher.stitch_homo((obj_img, ref_img), sys_config)
        if homomat is None:
            continue
        dx, dy = check_relative_pos_side_back(homomat, obj_img.shape[:2])
        tmp_n_ref.append(n_ref)
        tmp_dy.append(dy)
        tmp_dx.append(dx)
        tmp_hmat.append(homomat)
        print("the dx, and dy values 2ed", idx, n_ref, dx, dy, dy_threshold_2)
        if abs(dx) < dx_threshold_2 and abs(dy) < dy_threshold_2:
            print("the dx, and dy values final accepted", idx, n_ref, dx, dy, dy_threshold_2)
            find_neighbor = True  # for fine tune
            return find_neighbor, tmp_dicts
        elif (((-1 - delta) <= dx <= (-1 + delta)) and ((-1 - delta) <= dy <= (-1 + delta))) \
                or (abs(dx) > dx_threshold_1) or (abs(tmp_dy[-1]) > abs(tmp_dy[-2])):

            abs_dx = np.abs(np.array(tmp_dx))
            abs_dy = np.abs(np.array(tmp_dy))
            if min(abs_dx) <= dx_threshold_2 and min(abs_dy) <= dy_threshold_1:
                tmp_sum = 5 * abs_dx + abs_dy  # dx is more important
                min_dy = np.argmin(tmp_sum)
                find_neighbor, tmp_dicts = decision_from_dx_dy(sys_config, tmp_dicts, idx, tmp_dx[min_dy],
                                                               tmp_dy[min_dy], tmp_n_ref[min_dy], tmp_hmat[min_dy],
                                                               dy_threshold_2)
                return find_neighbor, tmp_dicts
            else:
                print('neighbor not found for negative dy!')
                homomat = None
                tmp_n_ref.append(n_ref)
                tmp_dy.append(dy)
                tmp_dx.append(dx)
                tmp_hmat.append(homomat)
                find_neighbor = True
                return find_neighbor, tmp_dicts
        elif dy > 0:
            print('dy change sign to positive')
            abs_dx = np.abs(np.array(tmp_dx))
            abs_dy = np.abs(np.array(tmp_dy))
            if min(abs_dx) <= dx_threshold_2 and min(abs_dy) <= dy_threshold_1:
                tmp_sum = 5 * abs_dx + abs_dy  # dx is more important
                min_dy = np.argmin(tmp_sum)
                find_neighbor, tmp_dicts = decision_from_dx_dy(sys_config, tmp_dicts, idx, tmp_dx[min_dy],
                                                               tmp_dy[min_dy], tmp_n_ref[min_dy], tmp_hmat[min_dy],
                                                               dy_threshold_2)
                return find_neighbor, tmp_dicts
            else:
                print('neighbor not found for negative dy!')
                homomat = None
                tmp_n_ref.append(n_ref)
                tmp_dy.append(dy)
                tmp_dx.append(dx)
                tmp_hmat.append(homomat)
                find_neighbor = True
                return find_neighbor, tmp_dicts

        elif i == n_check - 1:
            print('neighbor not found for negative dy!')
            homomat = None
            tmp_n_ref.append(n_ref)
            tmp_dy.append(dy)
            tmp_dx.append(dx)
            tmp_hmat.append(homomat)
            find_neighbor = True
            return find_neighbor, tmp_dicts


def positive_dy(detector, sys_config, tmp_dicts, obj_img, idx, n_ref, ref_img_paths, dy_sign, mask, dx, dy):
    """
    handel the situation when dy< 0 in find the best neighbor function.
    """
    print('the dy is positive !!')
    delta = float(sys_config['Neighbors']['tolerance'])  # tolerance for dy = -1.0 \pm delta
    dx_threshold_1 = float(sys_config['Neighbors']['dx_threshold_max'])  # 0.5
    dx_threshold_2 = float(sys_config['Neighbors']['dx_threshold_min'])  # 0.2
    dy_threshold_1 = float(sys_config['Neighbors']['dy_threshold_max'])  # 0.6
    dy_threshold_2 = float(sys_config['Neighbors']['dy_threshold_min'])  # 0.2
    tmp_dx = tmp_dicts['tmpdx']
    tmp_dy = tmp_dicts['tmpdy']
    tmp_n_ref = tmp_dicts['tmpnref']
    tmp_hmat = tmp_dicts['tmpmat']

    ref_total = len(ref_img_paths)
    n_check = 3
    for i in range(n_check):  # it is enough to shift 2 or 3 next neighbors
        n_ref = n_ref + int(dy_sign)  # check the sign
        if n_ref >= ref_total:
            print('REACH THE LAST REFERENCE IMG!!!')
            tmp_n_ref.append(n_ref)
            tmp_dy.append(dy)
            tmp_dx.append(dx)
            homomat = None
            tmp_hmat.append(homomat)
            find_neighbor = True  # for fine tune
            return find_neighbor, tmp_dicts
        ref_img_path = ref_img_paths[n_ref]
        ref_img = load_img(ref_img_path, mask=mask)  # np.load(ref_img_path)
        homomat = detector.stitcher.stitch_homo((obj_img, ref_img), sys_config)
        if homomat is None:
            continue
        dx, dy = check_relative_pos_side_back(homomat, obj_img.shape[:2])
        tmp_n_ref.append(n_ref)
        tmp_dy.append(dy)
        tmp_dx.append(dx)
        tmp_hmat.append(homomat)
        print("the dx, and dy values 2ed", idx, n_ref, dx, dy, dy_threshold_2)
        if abs(dx) < dx_threshold_2 and abs(dy) < dy_threshold_2:
            print("the dx, and dy values final accepted", idx, n_ref, dx, dy, dy_threshold_2)
            find_neighbor = True
            return find_neighbor, tmp_dicts
        elif (((-1 - delta) <= dx <= (-1 + delta)) and ((-1 - delta) <= dy <= (-1 + delta))) \
                or (abs(dx) > dx_threshold_1) or (abs(tmp_dy[-1]) > abs(tmp_dy[-2])):

            abs_dx = np.abs(np.array(tmp_dx))
            abs_dy = np.abs(np.array(tmp_dy))
            if min(abs_dx) <= dx_threshold_2 and min(abs_dy) <= dy_threshold_1:
                tmp_sum = 5 * abs_dx + abs_dy  # dx is more important
                min_dy = np.argmin(tmp_sum)
                find_neighbor, tmp_dicts = decision_from_dx_dy(sys_config, tmp_dicts, idx, tmp_dx[min_dy],
                                                               tmp_dy[min_dy], tmp_n_ref[min_dy], tmp_hmat[min_dy],
                                                               dy_threshold_2)
                return find_neighbor, tmp_dicts
            else:
                print('neighbor not found for negative dy!')
                img_dict = None
                tmp_n_ref.append(n_ref)
                tmp_dy.append(dy)
                tmp_dx.append(dx)
                tmp_hmat.append(homomat)
                find_neighbor = True
                return find_neighbor, tmp_dicts
        elif dy < 0:
            print('dy change sign to negative')
            abs_dx = np.abs(np.array(tmp_dx))
            abs_dy = np.abs(np.array(tmp_dy))
            if min(abs_dx) <= dx_threshold_2 and min(abs_dy) <= dy_threshold_1:
                tmp_sum = 5 * abs_dx + abs_dy  # dx is more important
                min_dy = np.argmin(tmp_sum)
                find_neighbor, tmp_dicts = decision_from_dx_dy(sys_config, tmp_dicts, idx, tmp_dx[min_dy],
                                                               tmp_dy[min_dy], tmp_n_ref[min_dy], tmp_hmat[min_dy],
                                                               dy_threshold_2)
                return find_neighbor, tmp_dicts
            else:
                print('neighbor not found for negative dy!')
                homomat = None
                tmp_n_ref.append(n_ref)
                tmp_dy.append(dy)
                tmp_dx.append(dx)
                tmp_hmat.append(homomat)
                find_neighbor = True
                return find_neighbor, tmp_dicts

        elif i == n_check - 1:
            print('neighbor not found for negative dy!')
            homomat = None
            tmp_n_ref.append(n_ref)
            tmp_dy.append(dy)
            tmp_dx.append(dx)
            tmp_hmat.append(homomat)
            find_neighbor = True
            return find_neighbor, tmp_dicts


def find_best_neighbor(detector, homomat, obj_img, id_obj, id_ref, ref_img_paths, table, sys_config, dy_sign, mask, num_comparison = 20):
    idx = id_obj
    n_ref = id_ref
    ref_total = len(ref_img_paths)
    delta = float(sys_config['Neighbors']['tolerance'])  # tolerance for dy = -1.0 \pm delta
    dx_threshold_1 = float(sys_config['Neighbors']['dx_threshold_max'])  # 0.5
    dx_threshold_2 = float(sys_config['Neighbors']['dx_threshold_min'])  # 0.2
    dy_threshold_1 = float(sys_config['Neighbors']['dy_threshold_max'])  # 0.6
    dy_threshold_2 = float(sys_config['Neighbors']['dy_threshold_min'])  # 0.2
    n_neighbors = int(sys_config['Neighbors']['n_check'])
    print("the Neighbors read from config", dx_threshold_1, dy_threshold_2, n_neighbors)
    #table[idx, 3] = n_ref
    table[idx][3] = n_ref
    tmp_n_ref = []
    tmp_dy = []
    tmp_dx = []
    tmp_hmat = []
    dx, dy = check_relative_pos_side_back(homomat, obj_img.shape[:2])
    #table[idx, 4] = dx
    #table[idx, 5] = dy
    table[idx][4] = dx
    table[idx][5] = dy
    tmp_n_ref.append(n_ref)
    tmp_dy.append(dy)
    tmp_dx.append(dx)
    tmp_hmat.append(homomat)
    print("the dx, and dy values 1st", idx, n_ref, dx, dy)
    find_neighbor = False
    tmp_cycles = 0
    tmp_dicts = {
        'tmpdx': tmp_dx,
        'tmpdy': tmp_dy,
        'tmpnref': tmp_n_ref,
        'tmpmat': tmp_hmat
    }

    # find_neighbor = False
    while not find_neighbor:
        tmp_cycles += 1
        dx = tmp_dicts['tmpdx'][-1]
        dy = tmp_dicts['tmpdy'][-1]
        n_ref = tmp_dicts['tmpnref'][-1]
        homomat = tmp_dicts['tmpmat'][-1]
        print('while loop number', tmp_cycles)
        if ((-1 - delta) <= dx <= (-1 + delta) and (-1 - delta) <= dy <= (-1. + delta)) or (abs(dx) > dx_threshold_1):
            find_neighbor, tmp_dicts = stitch_failed(detector, sys_config, tmp_dicts, obj_img,
                                                     idx, n_ref, ref_img_paths, dy_sign, mask, dx, dy, num_comparison)
            print('The dx xy after stitch_failed\n', tmp_dicts['tmpdx'][-1], tmp_dicts['tmpdy'][-1] )
           # break
        elif dy <= -1 * dy_threshold_2:
            print('dy < -1 ')
            find_neighbor, tmp_dicts = negative_dy(detector, sys_config, tmp_dicts, obj_img,
                                                   idx, n_ref, ref_img_paths, dy_sign, mask, dx, dy)
            #break
        elif dy >= dy_threshold_2:
            find_neighbor, tmp_dicts = positive_dy(detector, sys_config, tmp_dicts, obj_img,
                                                   idx, n_ref, ref_img_paths, dy_sign, mask, dx, dy)
            #break

        elif abs(dx) <= dx_threshold_2:
            print("the dx, and dy values final accepted", idx, n_ref, dx, dy, dy_threshold_2)
            find_neighbor = True
            #break
        else:
            table[idx][6] = n_ref
            table[idx][7] = dx
            table[idx][8] = dy
            table[idx][9] = dy_threshold_2
            #table[idx, 6] = n_ref
            #table[idx, 7] = dx
            #table[idx, 8] = dy
            #table[idx, 9] = dy_threshold_2
            homomat = None
            return table, homomat
    #table[idx, 6] = tmp_dicts['tmpnref'][-1]
    #table[idx, 7] = tmp_dicts['tmpdx'][-1]
    #table[idx, 8] = tmp_dicts['tmpdy'][-1]
    #table[idx, 9] = dy_threshold_2
    table[idx][6] = tmp_dicts['tmpnref'][-1]
    table[idx][7] = tmp_dicts['tmpdx'][-1]
    table[idx][8] = tmp_dicts['tmpdy'][-1]
    table[idx][9] = dy_threshold_2
    homomat = tmp_dicts['tmpmat'][-1]
    #if homomat is not None:
    best_ref = tmp_dicts['tmpnref'][-1]
   # else:
   #     best_ref = None
    return table, homomat, best_ref # df_arr_dy


# def find_best_neighbor_jry(num_comparison):
#     '''jry: newly added on July 20th'''
#     arr_dy = np.array([0 for _ in range(num_comparison)])
#     obj_insert_pos = [0]
#     gps = load_gps(os.path.join('', 'gps_measures.csv'))
#     ref = load_gps(os.path.join('', 'gps_measures.csv'))
#     for i in range(gps.shape[0]):
#         curr_lon = gps.loc[i, 'lon']
#         curr_lat = gps.loc[i, 'lat']
#         if i == 0:
#             obj_insert_pos.append(0)
#         else:
#             obj_insert_pos.append(obj_insert_pos[i - 1])
#         for j in range(len()):
#             find_neighbor = False
#             while not find_neighbor:
#                 if curr_lon < ref.loc[obj_insert_pos, 'lon'] and curr_lat < ref.loc[obj_insert_pos, 'lon']:
#                     obj_insert_pos[i] += 1
#                 else:
#                     while calculate_gps_distance((curr_lon, curr_lat), (ref.loc[obj_insert_pos, 'lon'], ref.loc[obj_insert_pos, 'lon']))[obj_insert_pos]\
#                             < calculate_gps_distance()[min(0, obj_insert_pos - 1)]:
#                         obj_insert_pos[i] += 1
#         find_neighbor = True
#         if i == 0:
#             arr_dy = np.array([dy for _ in range(num_comparison)])
#         else:
#             np.append(arr_dy, [dy in _ range(num_comparison)])
#     df_arr_dy = pd.DataFrame(arr_dy)
#     df_arr_dy.columns = ['dy_obj_{}'.format(str(i)) for i in range(- num_comparison, num_comparison + 1)]
#     return df_arr_dy


def find_best_neighbor_9Jyly(detector, img_dict, obj_img, id_obj, id_ref, ref_img_paths, table, sys_config, dy_sign,
                             mask):
    idx = id_obj
    n_ref = id_ref
    ref_total = len(ref_img_paths)
    delta = float(sys_config['Neighbors']['tolerance'])  # tolerance for dy = -1.0 \pm delta
    dx_threshold_1 = float(sys_config['Neighbors']['dx_threshold_max'])  # 0.5
    dx_threshold_2 = float(sys_config['Neighbors']['dx_threshold_min'])  # 0.2
    dy_threshold_1 = float(sys_config['Neighbors']['dy_threshold_max'])  # 0.6
    dy_threshold_2 = float(sys_config['Neighbors']['dy_threshold_min'])  # 0.2
    n_neighbors = int(sys_config['Neighbors']['n_check'])
    print("the Neighbors read from config", dx_threshold_1, dy_threshold_2, n_neighbors)
    table[idx, 3] = n_ref
    tmp_n_ref = []
    tmp_dy = []
    tmp_dx = []
    tmp_img_dict = []
    dx, dy = check_relative_pos_side_back(img_dict['homomtx'], obj_img.shape[:2])
    table[idx, 4] = dx
    table[idx, 5] = dy
    tmp_n_ref.append(n_ref)
    tmp_dy.append(abs(dy))
    tmp_dx.append(abs(dx))
    tmp_img_dict.append(img_dict)
    print("the dx, and dy values 1st", idx, n_ref, dx, dy)
    find_neighbor = False
    tmp_cycles = 0
    while not find_neighbor:
        tmp_cycles += 1
        print('while loop number', tmp_cycles)
        if ((-1 - delta) <= dx <= (-1 + delta) and (-1 - delta) <= dy <= (-1. + delta)) or (abs(dx) > dx_threshold_1):
            tmp_dicts = {'tmpdx': tmp_dx,
                         'tmpdy': tmp_dy,
                         'tmpnref': tmp_n_ref,
                         'tmpimgdict': tmp_img_dict
                         }
            neighbor_array = generate_neighbor_array(n_neighbors, n_ref, ref_total, dy_sign)
            print('enter here', idx, n_ref, dx, dy, dy_threshold_2, tmp_cycles)
            n_ref_0 = n_ref
            for i in neighbor_array:
                n_ref = n_ref_0 + int(i * dy_sign)
                ref_img_path = ref_img_paths[n_ref]
                ref_img = load_img(ref_img_path, mask=mask)  # np.load(ref_img_path)
                img_dict = detector.stitcher.stitch_ref((obj_img, ref_img), sys_config)
                if img_dict is None:
                    continue
                dx, dy = check_relative_pos_side_back(img_dict['homomtx'], obj_img.shape[:2])
                print("the dx, and dy values for failed run 1st", idx, n_ref, dx, dy, dy_threshold_2, tmp_cycles)
                if abs(dx) < dx_threshold_2 and abs(dy) < dy_threshold_2:
                    print("the dx, and dy values final accepted", idx, n_ref, dx, dy, dy_threshold_2, tmp_cycles)
                    find_neighbor = True
                    break
                elif abs(dx) < dx_threshold_2 and abs(dy) < dy_threshold_1:
                    print("the dx, and dy values for failed run 2ed", idx, n_ref, dx, dy, dy_threshold_2, tmp_cycles)
                    dy_threshold_2 += 0.1  #
                    tmp_n_ref.append(n_ref)
                    tmp_dy.append(abs(dy))
                    tmp_dx.append(abs(dx))
                    tmp_img_dict.append(img_dict)
                    break
                elif i == neighbor_array[-1]:
                    print('Always dx=dy=-1!')
                    table[idx, 6] = n_ref
                    table[idx, 7] = abs(dx)
                    table[idx, 8] = abs(dy)
                    table[idx, 9] = dy_threshold_2
                    img_dict = None
                    return table, img_dict
        elif dy <= -1 * dy_threshold_2:
            print('dy < -1 ')
            for i in range(1, 4, 1):
                n_ref = n_ref - int(dy_sign)  # check the sign
                if n_ref >= ref_total:
                    print('REACH THE LAST REFERENCE IMG!!!')
                    table[idx, 6] = n_ref
                    table[idx, 7] = abs(dx)
                    table[idx, 8] = abs(dy)
                    table[idx, 9] = dy_threshold_2
                    img_dict = None
                    return table, img_dict
                    # n_ref = n_ref % ref_total
                ref_img_path = ref_img_paths[n_ref]
                ref_img = load_img(ref_img_path, mask=mask)  # np.load(ref_img_path)
                img_dict = detector.stitcher.stitch_ref((obj_img, ref_img), sys_config)
                if img_dict is None:
                    continue
                dx, dy = check_relative_pos_side_back(img_dict['homomtx'], obj_img.shape[:2])
                tmp_n_ref.append(n_ref)
                tmp_dy.append(abs(dy))
                tmp_dx.append(abs(dx))
                tmp_img_dict.append(img_dict)
                print("the dx, and dy values 2ed", idx, n_ref, dx, dy, dy_threshold_2, tmp_cycles)
                if abs(dx) < dx_threshold_2 and abs(dy) < dy_threshold_2:
                    print("the dx, and dy values final accepted", idx, n_ref, dx, dy, dy_threshold_2, tmp_cycles)
                    find_neighbor = True
                    break
                elif (((-1 - delta) <= dx <= (-1 + delta)) and ((-1 - delta) <= dy <= (-1 + delta))) \
                        or (abs(dx) > dx_threshold_1):
                    if min(np.array(tmp_dx)) <= dx_threshold_2 and min(np.array(tmp_dy)) <= dy_threshold_1:
                        tmp_sum = 5 * np.array(tmp_dx) + np.array(tmp_dy)  # dx is more important
                        min_dy = np.argmin(np.array(tmp_sum))
                        print("the tmp_sum and min_dy", tmp_sum, min_dy)
                        table, img_dict = final_decision_dx_dy(sys_config, table, idx, tmp_dx[min_dy], tmp_dy[min_dy],
                                                               tmp_n_ref[min_dy], tmp_img_dict[min_dy], dy_threshold_2)
                        return table, img_dict
                    else:
                        break
                elif dy > 0:
                    print('the 1st tmpnref and tmpdy are', tmp_n_ref, tmp_dy, tmp_dx)
                    tmp_sum = 5 * np.array(tmp_dx) + np.array(tmp_dy)  # dx is more important
                    min_dy = np.argmin(np.array(tmp_sum))
                    print("the tmp_sum and min_dy", tmp_sum, min_dy)
                    table, img_dict = final_decision_dx_dy(sys_config, table, idx, tmp_dx[min_dy], tmp_dy[min_dy],
                                                           tmp_n_ref[min_dy], tmp_img_dict[min_dy], dy_threshold_2)
                    return table, img_dict
        elif dy >= dy_threshold_2:
            print('dy > 1 ')
            for i in range(1, 4, 1):
                n_ref = n_ref + int(dy_sign)  # check the sign
                if n_ref >= ref_total:
                    print('REACH THE LAST REFERENCE IMG!!!')
                    table[idx, 6] = n_ref
                    table[idx, 7] = abs(dx)
                    table[idx, 8] = abs(dy)
                    table[idx, 9] = dy_threshold_2
                    img_dict = None
                    return table, img_dict
                ref_img_path = ref_img_paths[n_ref]
                ref_img = load_img(ref_img_path, mask=mask)  # np.load(ref_img_path)
                img_dict = detector.stitcher.stitch_ref((obj_img, ref_img), sys_config)
                if img_dict is None:
                    continue
                dx, dy = check_relative_pos_side_back(img_dict['homomtx'], obj_img.shape[:2])
                tmp_dx.append(abs(dx))
                tmp_dy.append(abs(dy))
                tmp_n_ref.append(n_ref)
                tmp_img_dict.append(img_dict)
                print("the dx, and dy values 2ed", idx, n_ref, dx, dy, dy_threshold_2, tmp_cycles)
                if abs(dx) < dx_threshold_2 and abs(dy) < dy_threshold_2:
                    print("the dx, and dy values final accepted", idx, n_ref, dx, dy, dy_threshold_2, tmp_cycles)
                    find_neighbor = True
                    break
                elif (((-1 - delta) <= dx <= (-1 + delta)) and ((-1 - delta) <= dy <= (-1 + delta))) \
                        or (abs(dx) > dx_threshold_1):
                    if min(np.array(tmp_dx)) <= dx_threshold_2 and min(np.array(tmp_dy)) <= dy_threshold_1:
                        tmp_sum = 5 * np.array(tmp_dx) + np.array(tmp_dy)  # dx is more important
                        min_dy = np.argmin(np.array(tmp_sum))
                        table, img_dict = final_decision_dx_dy(sys_config, table, idx, tmp_dx[min_dy], tmp_dy[min_dy],
                                                               tmp_n_ref[min_dy], tmp_img_dict[min_dy], dy_threshold_2)
                        return table, img_dict
                    else:
                        break
                elif dy < 0:
                    print('the 2nd tmpnref and tmpdy are', tmp_n_ref, tmp_dy, tmp_dx)
                    tmp_sum = 5 * np.array(tmp_dx) + np.array(tmp_dy)  # dx is more important
                    min_dy = np.argmin(np.array(tmp_sum))
                    table, img_dict = final_decision_dx_dy(sys_config, table, idx, tmp_dx[min_dy], tmp_dy[min_dy],
                                                           tmp_n_ref[min_dy], tmp_img_dict[min_dy], dy_threshold_2)
                    return table, img_dict

        elif abs(dx) <= dx_threshold_2:
            print("the dx, and dy values final accepted", idx, n_ref, dx, dy, dy_threshold_2)
            find_neighbor = True
            break
        else:
            table[idx, 6] = n_ref
            table[idx, 7] = abs(dx)
            table[idx, 8] = abs(dy)
            table[idx, 9] = dy_threshold_2
            img_dict = None
            return table, img_dict
    table[idx, 6] = n_ref
    table[idx, 7] = abs(dx)
    table[idx, 8] = abs(dy)
    table[idx, 9] = dy_threshold_2
    if img_dict is not None:
        img_dict['best_ref'] = n_ref
    return table, img_dict


def find_best_neighbor_old(detector, tmp_img_dict, obj_img, id_obj, id_ref, ref_img_paths, table, sys_config, dy_sign,
                           mask):
    idx = id_obj
    n_ref = id_ref
    DX_THRESHOLD_1 = 5.0
    DX_THRESHOLD_2 = 0.2
    DY_THRESHOLD_1 = 0.6
    DY_THRESHOLD_2 = 0.2
    MAX_CYCLE = 1  # maximum cycles before DY_THRESHOLD + 0.1
    table[idx, 3] = n_ref
    tmp_n_ref = []
    tmp_dy = []
    tmp_dx = []
    tmp_dy = []
    dx, dy = check_relative_pos_side_back(tmp_img_dict['homomtx'], obj_img.shape[:2])
    tmp_dx.append(dx)
    tmp_dy.append(dy)
    table[idx, 4] = dx
    table[idx, 5] = dy
    tmp_n_ref.append(n_ref)
    tmp_dy.append(abs(dy))
    print("the dx, and dy values 1st", idx, n_ref, dx, dy)
    find_neighbor = False
    n_cycles = 0
    tmp_count = 0
    tmp_cycles = 0
    while not find_neighbor:
        tmp_cycles += 1
        print('while loop number', tmp_cycles)
        if dx == -1.0 and dy == -1.0:
            print('enter here', idx, n_ref, dx, dy, DY_THRESHOLD_2, n_cycles)
            n_ref_0 = n_ref
            for i in [1, -1, 2, -2]:  # try n_ref-2 to n_ref+2
                n_ref = n_ref_0 + int(i * dy_sign)
                ref_img_path = ref_img_paths[n_ref]
                ref_img = load_img(ref_img_path, mask=mask)  # np.load(ref_img_path)
                tmp_img_dict = detector.stitcher.stitch_ref((obj_img, ref_img), sys_config)
                if tmp_img_dict is None:
                    continue
                dx, dy = check_relative_pos_side_back(tmp_img_dict['homomtx'], obj_img.shape[:2])
                print("the dx, and dy values for failed run 1st", idx, n_ref, dx, dy, DY_THRESHOLD_2, tmp_count)
                if abs(dx) < DX_THRESHOLD_2 and abs(dy) < DY_THRESHOLD_2:
                    print("the dx, and dy values final accepted", idx, n_ref, dx, dy, DY_THRESHOLD_2, tmp_count)
                    find_neighbor = True
                    break
                # elif abs(dx) < DX_THRESHOLD_2:
                #    print("the dx, and dy values for failed run 2ed", idx, n_ref, dx, dy, DY_THRESHOLD_2, n_cycles)
                #    DY_THRESHOLD_2 += 0.1  # modify the logic later
                #    break
                elif abs(dx) < DX_THRESHOLD_2 and abs(dy) < DY_THRESHOLD_1:
                    print("the dx, and dy values for failed run 2ed", idx, n_ref, dx, dy, DY_THRESHOLD_2, n_cycles)
                    DY_THRESHOLD_2 += 0.2  #
                    tmp_n_ref.append(n_ref)
                    tmp_dy.append(abs(dy))
                    break
                elif tmp_count >= MAX_CYCLE:
                    print("too many cycles when dx=dy=-1")
                    DY_THRESHOLD_2 += 0.2  # modify the logic later
                    # find_neighbor = True
                    # dy = None
                    # tmp_img_dict = None
                    # continue
                    break
                else:
                    tmp_count += 1
        # elif abs(dx) <= DX_THRESHOLD_1:  # does this make sense?
        elif n_cycles < MAX_CYCLE:
            # if n_cycles <= MAX_CYCLE:
            print('the number of cycles', n_cycles, MAX_CYCLE)
            if dy <= -1 * DY_THRESHOLD_2:
                n_ref = n_ref - int(1 * dy_sign)  # check the sign
                ref_img_path = ref_img_paths[n_ref]
                ref_img = load_img(ref_img_path, mask=mask)  # np.load(ref_img_path)
                tmp_img_dict = detector.stitcher.stitch_ref((obj_img, ref_img), sys_config)
                if tmp_img_dict is None:
                    continue
                dx, dy = check_relative_pos_side_back(tmp_img_dict['homomtx'], obj_img.shape[:2])
                tmp_n_ref.append(n_ref)
                tmp_dy.append(abs(dy))
                n_cycles += 1
                print("the dx, and dy values 2ed", idx, n_ref, dx, dy, DY_THRESHOLD_2, n_cycles)
            elif dy >= DY_THRESHOLD_2:
                n_ref = n_ref + int(1 * dy_sign)  # check the sign
                ref_img_path = ref_img_paths[n_ref]
                ref_img = load_img(ref_img_path, mask=mask)  # np.load(ref_img_path)
                tmp_img_dict = detector.stitcher.stitch_ref((obj_img, ref_img), sys_config)
                if tmp_img_dict is None:
                    continue
                dx, dy = check_relative_pos_side_back(tmp_img_dict['homomtx'], obj_img.shape[:2])
                tmp_dx.append(dx)
                tmp_dy.append(dy)
                tmp_n_ref.append(n_ref)
                tmp_dy.append(abs(dy))
                n_cycles += 1
                print("the dx, and dy values 2ed", idx, n_ref, dx, dy, DY_THRESHOLD_2, n_cycles)
            elif abs(dx) <= DX_THRESHOLD_2:
                print("the dx, and dy values final accepted", idx, n_ref, dx, dy, DY_THRESHOLD_2)
                find_neighbor = True
                break
            else:
                n_cycles = MAX_CYCLE + 1
            #    continue
        elif DY_THRESHOLD_2 <= DY_THRESHOLD_1:
            DY_THRESHOLD_2 += 0.2
            print('the tmpnref and tmpdy are', tmp_n_ref, tmp_dy)
            n_ref = tmp_n_ref[np.argmin(np.array(tmp_dy))]
            print("the np argmin", np.argmin(np.array(tmp_dy)))
            # n_ref = n_ref_fix
            ref_img_path = ref_img_paths[n_ref]
            ref_img = load_img(ref_img_path, mask=mask)  # np.load(ref_img_path)
            tmp_img_dict = detector.stitcher.stitch_ref((obj_img, ref_img), sys_config)
            if tmp_img_dict is None:
                continue
            dx, dy = check_relative_pos_side_back(tmp_img_dict['homomtx'], obj_img.shape[:2])
            n_cycles = 0
            print("the dx, and dy values with new threshold", idx, n_ref, dx, dy, DY_THRESHOLD_2)
        else:
            dy = None
            tmp_img_dict = None
            break
        # else:
        #    dy = None  ## starting neighbor is too bad!!
        #    tmp_img_dict = None
        #    break
    table[idx, 6] = n_ref
    table[idx, 7] = dx
    table[idx, 8] = dy
    table[idx, 9] = DY_THRESHOLD_2
    table[idx, 11] = tmp_cycles
    if tmp_img_dict is not None:
        tmp_img_dict['best_ref'] = n_ref
    df = pd.DataFrame(data=table,
                      columns=['obj', 'ref_1', 'delta', 'ref_2', 'dx', 'dy', 'ref_new', 'dx_final', 'dy_final',
                               'thresh_y', 'time(s)', 'ncycles', 'time_cpp'])
    df.to_csv(os.path.join(detector.output_path, "check_dy_neighbor" + ".csv"))
    return tmp_img_dict


def offline_process(cfg_path: AnyStr, start_obj, delta, mask, num_comparison = 20):
    """offline process

    Args:
        cfg_path:
        ref_img_dir:
        curr_img_dir:

    Returns:

    """
    cfg_sys = fpp.ini_to_dict(cfg_path)
    detector = AirportRunwayFFCmp(**cfg_sys["Detector"])
    curr_paths = generate_folder_path(cfg_sys["General"]["curr_dir"])
    print('the current path', curr_paths['image'])
    curr_raw_gps = load_gps(curr_paths["sensor_data"])
    ref_paths = generate_folder_path(cfg_sys["General"]["ref_dir"])
    ref_raw_gps = load_gps(ref_paths["sensor_data"])
    curr_img_paths = glob.glob(os.path.join(str(curr_paths['image']), cfg_sys['Camera']['serial'], "*" + IMG_EXT))
    ref_img_paths = glob.glob(os.path.join(str(ref_paths['image']), cfg_sys['Camera']['serial'], "*" + IMG_EXT))
    curr_img_paths.sort()
    ref_img_paths.sort()
    dy_sign = find_dy_sign(detector, cfg_sys, ref_img_paths, nref=2)
    number_curr_img = len(curr_img_paths)
    ref_ts = [float(os.path.splitext(os.path.split(x)[-1])[0]) for x in ref_img_paths]
    ref_ts = np.asarray(ref_ts).reshape(-1, 1)  # become one column nx1 shape

    ref_img_gps = generate_metadata(ref_ts, ref_raw_gps)
    ref_img_gps = pd.DataFrame(ref_img_gps, columns=["ts", "lat", "lon"])
    curr_gps = {}
    delta_lat = 0  # not implemented yet
    delta_lon = 0  # not implemented yet
    neighbor_info = []
    #table = np.zeros((number_curr_img, 12))

    tmp_table = np.array([[0 for i in range(11)]], dtype=float)
    table = np.array([[0 for i in range(11)]], dtype=float)
    save_jpg = False
    for id_obj, curr_img_path in enumerate(curr_img_paths[:10]):
        #        if save_jpg:
        #            curr_img = load_img(curr_img_path, mask=None)
        #            print('saving', id_obj)
        #            path = '/media/flyinstinct/Backup/data_Lyon_1July/combined/01-07-2020/2020_07_01_16_34_20A_40_O_PC1/img/19195148_jpg/obj_'
        #            cv2.imwrite(path + str(id_obj) + '.jpg', curr_img)
        #        #if id_obj == 93:
        #        else:

        if id_obj >= start_obj:  # the first object img may have no reference
            curr_img = load_img(curr_img_path, mask=None)
            curr_ts = float(os.path.splitext(os.path.split(curr_img_path)[-1])[0])
            curr_ts = np.asarray(curr_ts).reshape(-1, 1)
            curr_img_gps = generate_metadata(curr_ts, curr_raw_gps)
            curr_gps["lat"] = curr_img_gps[0, 1] + delta_lat
            curr_gps["lon"] = curr_img_gps[0, 2] + delta_lon

            if READ_NEIGHBOR:  # read manually chosen neighbor
                neighbor_path = '/home/flyinstinct/Documents/output_test/2020_07_02_14_32_41/check_dy_neighbor.csv'
                neighbor_read = pd.read_csv(neighbor_path)
                id_ref = int(neighbor_read['ref_new'].loc[id_obj])
                ref_img = load_img(ref_img_paths[id_ref], mask=mask)
                img_dict = detector.stitcher.stitch_ref((curr_img, ref_img), cfg_sys)
                stitch_cmp(detector, img_dict, curr_img, id_obj, id_ref, cfg_sys)
            else:
                neighbor_name = find_best_gps_neighbor(curr_gps, ref_img_gps,
                                                       dist_threshold=int(cfg_sys["General"]["preload_radius"]))
                # neighbor_name = float(neighbor_1st[id_obj,1])
                table = np.append(table, tmp_table, axis=0)
                id_ref = np.argmax(ref_ts == neighbor_name)
                #table[id_obj, 0] = id_obj
                #table[id_obj, 1] = id_ref
                #table[id_obj, 2] = delta
                table[id_obj][0] = id_obj
                table[id_obj][1] = id_ref
                table[id_obj][2] = delta

                neighbor_info.append([curr_ts, neighbor_name, id_obj, id_ref])
                print("id_ref before correction", id_obj, id_ref)
                id_ref = id_ref + delta  # take care of gps correction
                id_ref_0 = id_ref
                print("id_ref after correction", id_obj, id_ref)
                if neighbor_name is None:
                    detector.logger.info("No neighbor is found for {}".format(curr_img_path))
                    continue
                # print('the reference_img_path is \n', os.path.join(ref_img_dir, str(neighbor_name) + IMG_EXT) )
                ref_img = load_img(ref_img_paths[id_ref], mask=mask)
                #
                tic = time.time()
                homomat = detector.stitcher.stitch_homo((curr_img, ref_img), cfg_sys)
                tac = time.time()
                print("the time spent on homomatrix is:" , tac-tic)
                #tmp_img_dict = detector.stitcher.stitch_ref((curr_img, ref_img), cfg_sys)
                if homomat is None:
                    detector.logger.info("No neighbor is found for {}".format(curr_img_path))
                    continue
                else:
                    tic = time.time()
                    table, homomat, best_ref = find_best_neighbor(detector, homomat, curr_img, id_obj, id_ref,
                                                         ref_img_paths, table, cfg_sys, dy_sign, mask, num_comparison)
                    tac = time.time()
                    table[id_obj][10] = tac - tic
                if homomat is not None:
                    ref_img = load_img(ref_img_paths[best_ref], mask=mask)
                    img_dict = detector.stitcher.stitch_wrap((curr_img, ref_img), homomat)
                    #print('the img_dict in offline', img_dict)
                    id_ref = best_ref
                    delta = delta + id_ref - id_ref_0
                    #table[id_obj, 11] = tac - tic
                    #table[id_obj][11] = id_obj
                else:
                    detector.logger.info("No neighbor is found for {}".format(curr_img_path))
                    continue
                if img_dict is None:
                    detector.logger.info("No neighbor is found for {}".format(curr_img_path))
                    continue
                else:
                    stitch_cmp(detector, img_dict, curr_img, id_obj, id_ref, cfg_sys)
    df = pd.DataFrame(data=table,
                      columns=['obj', 'ref_1', 'delta', 'ref_2', 'dx', 'dy', 'ref_new', 'dx_final', 'dy_final',
                               'thresh_y', 'time(s)'])
    df.to_csv(os.path.join(detector.output_path, "check_dy_neighbor" + ".csv"), index=False)
    neighbor_arr = np.asarray(neighbor_info).reshape(-1, 4)
    np.savetxt(os.path.join(detector.output_path, "neighbor_info.txt"), neighbor_arr)
    log_file = '/home/flyinstinct/Documents/pycharm_logs/log'
    os.system('mv ' + log_file + ' ' + detector.output_path)
    #    print("the output path is", detector.output_path)
    #    print('the split of detector out put path', detector.output_path.split('/')[-1])
    if SAVE_GOOGLE_DRIVE:
        my_drive_dir = '/home/flyinstinct/Desktop/google_drive'
        # here to mount google-drive with my local dir
        os.system('google-drive-ocamlfuse' + ' ' + my_drive_dir)
        # the directory where you want to rsync to google drive
        source_dir = detector.output_path
        # the directory where you linked to the google_drive, I recommend you to
        # keep the tree of the output so you do not mess up the run from different
        # days.
        target_dir = my_drive_dir + '/test_rsync/' + detector.output_path.split('/')[-1]
        os.system('rsync --update -zavh' + ' ' + source_dir + ' ' + target_dir)
        # unmount after rsync if you want
        os.system('fusermount -u' + ' ' + my_drive_dir)


def offline_process_jry(cfg_path: AnyStr, start_obj, delta, mask, save_dir = '/home/flyinstinct/Documents/output_Lyon_interpolated', num_comparison = 6):
    """offline process

    Args:
        cfg_path:
        ref_img_dir:
        curr_img_dir:

    Returns:

    """
    cfg_sys = fpp.ini_to_dict(cfg_path)
    detector = AirportRunwayFFCmp(**cfg_sys["Detector"])
    curr_paths = generate_folder_path(cfg_sys["General"]["curr_dir"])
    print('the current path', curr_paths['image'])
    curr_raw_gps = load_gps(curr_paths["sensor_data"])
    ref_paths = generate_folder_path(cfg_sys["General"]["ref_dir"])
    ref_raw_gps = load_gps(ref_paths["sensor_data"])
    curr_img_paths = glob.glob(os.path.join(str(curr_paths['image']), cfg_sys['Camera']['serial'], "*" + IMG_EXT))
    ref_img_paths = glob.glob(os.path.join(str(ref_paths['image']), cfg_sys['Camera']['serial'], "*" + IMG_EXT))
    curr_img_paths.sort()
    ref_img_paths.sort()
    dy_sign = find_dy_sign(detector, cfg_sys, ref_img_paths, nref=2)
    number_curr_img = len(curr_img_paths)
    ref_ts = [float(os.path.splitext(os.path.split(x)[-1])[0]) for x in ref_img_paths]
    ref_ts = np.asarray(ref_ts).reshape(-1, 1)  # become one column nx1 shape

    ref_img_gps = generate_metadata(ref_ts, ref_raw_gps)
    ref_img_gps = pd.DataFrame(ref_img_gps, columns=["ts", "lat", "lon"])
    curr_gps = {}
    delta_lat = 0  # not implemented yet
    delta_lon = 0  # not implemented yet
    neighbor_info = []
    #table = np.zeros((number_curr_img, 12))

    tmp_table = np.array([[0 for i in range(11)]], dtype=float)
    table = np.array([[0 for i in range(11)]], dtype=float)
    save_jpg = False
    for id_obj, curr_img_path in enumerate(curr_img_paths[:]):
        #        if save_jpg:
        #            curr_img = load_img(curr_img_path, mask=None)
        #            print('saving', id_obj)
        #            path = '/media/flyinstinct/Backup/data_Lyon_1July/combined/01-07-2020/2020_07_01_16_34_20A_40_O_PC1/img/19195148_jpg/obj_'
        #            cv2.imwrite(path + str(id_obj) + '.jpg', curr_img)
        #        #if id_obj == 93:
        #        else:

        if id_obj >= start_obj:  # the first object img may have no reference
            curr_img = load_img(curr_img_path, mask=None)
            curr_ts = float(os.path.splitext(os.path.split(curr_img_path)[-1])[0])
            curr_ts = np.asarray(curr_ts).reshape(-1, 1)
            curr_img_gps = generate_metadata(curr_ts, curr_raw_gps)
            curr_gps["lat"] = curr_img_gps[0, 1] + delta_lat
            curr_gps["lon"] = curr_img_gps[0, 2] + delta_lon


            neighbor_name = find_best_gps_neighbor(curr_gps, ref_img_gps,
                                                   dist_threshold=int(cfg_sys["General"]["preload_radius"]))
            id_ref_0 = np.argmax(ref_ts == neighbor_name)
            for id_ref in generate_neighbor_array(num_comparison, id_ref_0, ref_img_gps.shape[0], dy_sign):
                if READ_NEIGHBOR:  # read manually chosen neighbor
                    neighbor_path = '/home/flyinstinct/Documents/output_test/2020_07_02_14_32_41/check_dy_neighbor.csv'
                    neighbor_read = pd.read_csv(neighbor_path)
                    # id_ref = int(neighbor_read['ref_new'].loc[id_obj])
                    ref_img = load_img(ref_img_paths[id_ref], mask=mask)
                    img_dict = detector.stitcher.stitch_ref((curr_img, ref_img), cfg_sys)
                    stitch_cmp(detector, img_dict, curr_img, id_obj, id_ref, cfg_sys)
                else:
                    # neighbor_name = find_best_gps_neighbor(curr_gps, ref_img_gps,
                    #                                        dist_threshold=int(cfg_sys["General"]["preload_radius"]))
                    # neighbor_name = float(neighbor_1st[id_obj,1])
                    table = np.append(table, tmp_table, axis=0)
                    # id_ref = np.argmax(ref_ts == neighbor_name)
                    #table[id_obj, 0] = id_obj
                    #table[id_obj, 1] = id_ref
                    #table[id_obj, 2] = delta
                    table[-1][0] = id_obj
                    table[-1][1] = id_ref_0
                    table[-1][2] = id_ref
                    table[-1][3] = delta


                    # neighbor_info.append([curr_ts, neighbor_name, id_obj, id_ref])
                    print("id_ref before correction", id_obj, id_ref)
                    # id_ref = id_ref + delta  # take care of gps correction
                    # id_ref_0 = id_ref
                    print("id_ref after correction", id_obj, id_ref)
                    if neighbor_name is None:
                        detector.logger.info("No neighbor is found for {}".format(curr_img_path))
                        continue
                    # print('the reference_img_path is \n', os.path.join(ref_img_dir, str(neighbor_name) + IMG_EXT) )
                    ref_img = load_img(ref_img_paths[id_ref], mask=mask)
                    #
                    tic = time.time()
                    homomat = detector.stitcher.stitch_homo((curr_img, ref_img), cfg_sys)
                    tac = time.time()
                    print("the time spent on homomatrix is:" , tac-tic)
                    #tmp_img_dict = detector.stitcher.stitch_ref((curr_img, ref_img), cfg_sys)
                    if homomat is None:
                        detector.logger.info("No neighbor is found for {}".format(curr_img_path))
                        continue
                    else:
                        tic = time.time()
                        # table, homomat, best_ref = find_best_neighbor(detector, homomat, curr_img, id_obj, id_ref,
                        #                                      ref_img_paths, table, cfg_sys, dy_sign, mask, num_comparison)
                        dx, dy = check_relative_pos_side_back(homomat, curr_img.shape[:2])
                        table[-1][4] = dx
                        table[-1][5] = dy
                        tac = time.time()
                        table[-1][10] = tac - tic
                    if homomat is not None:
                        # ref_img = load_img(ref_img_paths[best_ref], mask=mask)
                        ref_img = load_img(ref_img_paths[id_ref], mask=mask)
                        img_dict = detector.stitcher.stitch_wrap((curr_img, ref_img), homomat)
                        #print('the img_dict in offline', img_dict)
                        # id_ref = best_ref
                        # delta = delta + id_ref - id_ref_0
                        #table[id_obj, 11] = tac - tic
                        #table[id_obj][11] = id_obj
                    else:
                        detector.logger.info("No neighbor is found for {}".format(curr_img_path))
                        continue
                    if img_dict is None:
                        detector.logger.info("No neighbor is found for {}".format(curr_img_path))
                        continue
                    else:
                        stitch_cmp(detector, img_dict, curr_img, id_obj, id_ref, cfg_sys)
    # df = pd.DataFrame(data=table,
    #                   columns=['obj', 'ref_1', 'delta', 'ref_2', 'dx', 'dy', 'ref_new', 'dx_final', 'dy_final',
    #                            'thresh_y', 'time(s)'])
    # df.to_csv(os.path.join(detector.output_path, "check_dy_neighbor" + ".csv"), index=False)
    df_01 = pd.DataFrame(data = table[:, [i for i in range(6)]], columns = ['id_obj', 'neighbor_by_gps','id_ref_compared', 'delta', 'dx', 'dy'])
    df_01.drop(df_01.iloc[0, :], inplace = True)
    df_01.to_csv(os.path.join(detector.output_path, "check_dy_neighbor" + ".csv"), index=False)
    l_ref = list(ref_img_paths)
    # print('l_ref', l_ref)
    for j in df_01['id_obj'].unique():
        df_insert_pos = df_01[df_01['id_obj'] == j]
        df_insert_pos = df_insert_pos[abs(df_insert_pos['dy']) <= 0.7]['id_ref_compared']
        if len(df_insert_pos) >= 2:
            pos = int(df_insert_pos.to_list()[-1])
        elif len(df_insert_pos.shape) == 1:
            pos = int(df_insert_pos.to_list()[-1] + np.sign(df_01.loc[df_insert_pos.to_list()[-1], ['dy']]) * dy_sign)
        else:
            print('No ref matched to curr!!!!!')
            break
        # print('j =', j, curr_img_paths)
        val = list(curr_img_paths)[int(j)]
        # cv2.imwrite(os.path.join(save_dir, "{}_{}_1_curr_img.jpg".format(curr_img_name, ref_img_name)),
        #             load_img(curr_paths[id_obj]))
        l_ref.insert(pos, val)
        pos += 2
    # pd.to_csv(l_ref)
    print('l_ref', l_ref)
    name_interpolated = deepcopy(l_ref)
    count_curr = 0
    print('l_ref_2222', l_ref)
    for i in range(len(name_interpolated)):
        if str(name_interpolated[i].split('/')[-4]).endswith('_NO') == True:
            name_interpolated[i] = name_interpolated[i].split('/')[-1].replace('.npy', '.jpg')
            # print('i', i, len(l_ref), l_ref[i], l_ref[i].split('/'))
        elif str(name_interpolated[i].split('/')[-4]).endswith('_PC1') == True:
            name_interpolated[i] = name_interpolated[i].split('/')[-1].replace('.npy', '.jpg')
            start = i
            end = i
            # print('i', i, len(l_ref), l_ref[i], l_ref[i].split('/'))
            while start > 0 and l_ref[max(start, 0)].split('/')[-4].endswith('_PC1'):
                start -= 1
            while end < len(name_interpolated) - 1 and l_ref[min(end, len(name_interpolated) - 1)].split('/')[-4].endswith('_PC1'):
                end += 1
            if start == 0:
                name_interpolated[i] = str(float(name_interpolated[end][:-5].split('/')[-1]) - 0.1) + '_curr_{}'.format(str(count_curr)) + '.jpg'
            elif end == len(name_interpolated) - 1:
                name_interpolated[i] = str(float(name_interpolated[start][:-5].split('/')[-1]) + 0.1) + '_curr_{}'.format(str(count_curr)) + '.jpg'
            else:
                name_interpolated[i] = str((float(name_interpolated[start][:-5].split('/')[-1]) + float(name_interpolated[end][:-5].split('/')[-1])) / 2) + '_curr_{}'.format(str(count_curr)) + '.jpg'
            count_curr += 1
    print('name_interpolated', name_interpolated)
    # print(l_ref)

    # count_curr = 0
    # for i in range(len(l_ref)):
    #     # if len(str(l_ref[i].split('_curr_'))) > 1:
    #     if str(l_ref[i].split('/')[-4]).endswith(('_PC1')) == True:
    #         if i > 0 and i < len(l_ref) - 1:
    #             save_path_jpg = os.path.join(save_dir,
    #                              str((float(l_ref[i - 1].split('_curr_')[0].strip('.jpg')) + float(l_ref[i + 1].split('_curr_')[0].strip('.jpg'))) / 2)\
    #                              + '_curr_{}.jpg'.format(str(count_curr)))
    #         elif i == 0:
    #             save_path_jpg = os.path.join(save_dir,
    #                              str(float(l_ref[i + 1].split('_curr_')[0].strip('.jpg')) - 0.1) + '_curr_{}.jpg'.format(str(count_curr)))
    #         elif i == len(l_ref) - 1:
    #             save_path_jpg = os.path.join(save_dir,
    #                              str(float(l_ref[i - 1].split ('_curr_')[0].strip('.jpg')) + 0.1) + '_curr_{}.jpg'.format(str(count_curr)))
    #
    #         count_curr += 1
    #     # elif len(str(l_ref[i].split('_curr_'))) == 1:
    #     elif str(l_ref[i].split('/')[-4]).endswith(('_NO')) == True:

    #
    # print(l_ref)
    for i in range(len(name_interpolated)):
        save_path_jpg = os.path.join(save_dir, name_interpolated[i])
        img = load_img(l_ref[i])
        cv2.imwrite(save_path_jpg, img)



    neighbor_arr = np.asarray(neighbor_info).reshape(-1, 4)
    np.savetxt(os.path.join(detector.output_path, "neighbor_info.txt"), neighbor_arr)
    log_file = '/home/flyinstinct/Documents/pycharm_logs/log'
    os.system('mv ' + log_file + ' ' + detector.output_path)
    #    print("the output path is", detector.output_path)
    #    print('the split of detector out put path', detector.output_path.split('/')[-1])
    if SAVE_GOOGLE_DRIVE:
        my_drive_dir = '/home/flyinstinct/Desktop/google_drive'
        # here to mount google-drive with my local dir
        os.system('google-drive-ocamlfuse' + ' ' + my_drive_dir)
        # the directory where you want to rsync to google drive
        source_dir = detector.output_path
        # the directory where you linked to the google_drive, I recommend you to
        # keep the tree of the output so you do not mess up the run from different
        # days.
        target_dir = my_drive_dir + '/test_rsync/' + detector.output_path.split('/')[-1]
        os.system('rsync --update -zavh' + ' ' + source_dir + ' ' + target_dir)
        # unmount after rsync if you want
        os.system('fusermount -u' + ' ' + my_drive_dir)




def find_neighbor_from_GPS(cfg_path: AnyStr, ref_img_dir: AnyStr, curr_img_dir: AnyStr):
    """offline process

    Args:
        cfg_path:
        ref_img_dir:
        curr_img_dir:

    Returns:

    """
    cfg_sys = fpp.ini_to_dict(cfg_path)
    detector = AirportRunwayFFCmp(**cfg_sys["Detector"])
    curr_paths = generate_folder_path(cfg_sys["General"]["curr_dir"])
    curr_raw_gps = load_gps(curr_paths["sensor_data"])
    ref_paths = generate_folder_path(cfg_sys["General"]["ref_dir"])
    ref_raw_gps = load_gps(ref_paths["sensor_data"])
    curr_img_paths = glob.glob(os.path.join(curr_img_dir, "*" + IMG_EXT))
    ref_img_paths = glob.glob(os.path.join(ref_img_dir, "*" + IMG_EXT))
    curr_img_paths.sort()
    ref_img_paths.sort()
    number_curr_img = len(curr_img_paths)
    ref_ts = [float(os.path.splitext(os.path.split(x)[-1])[0]) for x in ref_img_paths]
    ref_ts = np.asarray(ref_ts).reshape(-1, 1)  # become one column nx1 shape

    ref_img_gps = generate_metadata(ref_ts, ref_raw_gps)
    ref_img_gps = pd.DataFrame(ref_img_gps, columns=["ts", "lat", "lon"])
    curr_gps = {}
    delta_lat = 0  ## not implemented yet
    delta_lon = 0
    neighbor_info = []
    start_obj = 0
    for id_obj, curr_img_path in enumerate(curr_img_paths):
        if id_obj >= start_obj:  ## the first object img may have no reference
            curr_img = load_img(curr_img_path)
            curr_ts = float(os.path.splitext(os.path.split(curr_img_path)[-1])[0])
            curr_ts = np.asarray(curr_ts).reshape(-1, 1)
            curr_img_gps = generate_metadata(curr_ts, curr_raw_gps)
            curr_gps["lat"] = curr_img_gps[0, 1] + delta_lat
            curr_gps["lon"] = curr_img_gps[0, 2] + delta_lon
            neighbor_name = find_best_gps_neighbor(curr_gps, ref_img_gps,
                                                   dist_threshold=int(cfg_sys["General"]["preload_radius"]))
            id_ref = np.argmax(ref_ts == neighbor_name)
            neighbor_info.append([curr_ts, neighbor_name, id_obj, id_ref])
    neighbor_arr = np.asarray(neighbor_info).reshape(-1, 4)
    np.savetxt("neighbor_from_GPS.txt", neighbor_arr)


def search_neighbors_manually(cfg_path, ref_dir, obj_dir, start_obj, PAIR_PATH):
    # ref_dir = os.path.join(REF_DIR,str(CAM_SERIAL))
    # obj_dir = os.path.join(OBJ_DIR,str(CAM_SERIAL))
    cfg_sys = fpp.ini_to_dict(cfg_path)
    detector = AirportRunwayFFCmp(**cfg_sys["Detector"])
    ref_name = glob.glob(os.path.join(ref_dir, "*" + IMG_EXT))
    ref_name.sort()
    obj_name = glob.glob(os.path.join(obj_dir, "*" + IMG_EXT))
    obj_name.sort()
    final_pair = np.zeros((len(obj_name), 4))
    full_id_ref = np.zeros(len(ref_name))
    for i in range(len(ref_name)):
        full_id_ref[i] = float(os.path.splitext(ref_name[i].split('/')[-1])[0])
    pair_info = np.loadtxt(PAIR_PATH)
    delta = 0
    nb_pair = pair_info.shape[0]
    count = 0
    for idx in range(nb_pair):
        if idx >= start_obj:
            final_pair[idx, 0] = idx
            n_ref = np.argmax(np.array(full_id_ref, dtype=float) == float(pair_info[idx, 1])) + delta
            final_pair[idx, 1] = n_ref
            n_ref_0 = n_ref
            obj_img_path = obj_name[idx]
            ref_img_path = ref_name[n_ref]
            ref_img = load_img(ref_img_path)  # np.load(ref_img_path).astype(np.uint8)
            obj_img = load_img(obj_img_path)  # np.load(obj_img_path).astype(np.uint8)
            h, w = obj_img.shape[0: 2]
            tmp_img_dict = detector.stitcher.stitch_ref((obj_img, ref_img), cfg_sys)
            Hmat = tmp_img_dict['homomtx']
            dx, dy = check_relative_pos_side_back(Hmat, (h, w))
            cv2.namedWindow("object", cv2.WINDOW_NORMAL)
            cv2.imshow("object", obj_img)
            cv2.resizeWindow("object", int(w * 0.2), int(h * 0.2))
            cv2.moveWindow("object", 1, 1)
            cv2.namedWindow("reference", cv2.WINDOW_NORMAL)
            cv2.imshow("reference", ref_img)
            cv2.resizeWindow("reference", int(w * 0.2), int(h * 0.2))
            cv2.moveWindow("reference", int(w * 0.2) + 2, 1)
            print('idx and n_ref 1st', idx, n_ref, dx, dy)
            print("press b for back one ref, n for next ref, s to select ref")
            print("press p for passing to the next obj")
            kb = cv2.waitKey(0)
            find_pair = False
            while not find_pair:
                if kb == ord("b"):
                    n_ref = n_ref - 1
                    ref_img_path = ref_name[n_ref]
                    ref_img = load_img(ref_img_path)  # np.load(ref_img_path)
                    tmp_img_dict = detector.stitcher.stitch_ref((obj_img, ref_img), cfg_sys)
                    Hmat = tmp_img_dict['homomtx']
                    dx, dy = check_relative_pos_side_back(Hmat, (h, w))
                    print('idx, n_ref, dx, dy after p', idx, n_ref, dx, dy)
                    cv2.namedWindow("reference", cv2.WINDOW_NORMAL)
                    cv2.imshow("reference", ref_img)
                    print("press b for back one ref, n for next ref, s to select ref")
                    print("press p for passing to the next obj")
                    kb = cv2.waitKey(0)
                elif kb == ord('n'):
                    n_ref = n_ref + 1
                    ref_img_path = ref_name[n_ref]
                    ref_img = load_img(ref_img_path)  # np.load(ref_img_path)
                    tmp_img_dict = detector.stitcher.stitch_ref((obj_img, ref_img), cfg_sys)
                    Hmat = tmp_img_dict['homomtx']
                    dx, dy = check_relative_pos_side_back(Hmat, (h, w))
                    print('idx, n_ref, dx, dy after n', idx, n_ref, dx, dy)
                    cv2.namedWindow("reference", cv2.WINDOW_NORMAL)
                    cv2.imshow("reference", ref_img)
                    print("press b for back one ref, n for next ref, s to select ref")
                    print("press p for passing to the next obj")
                    kb = cv2.waitKey(0)
                elif kb == ord("s"):
                    count += 1
                    tmp_img_dict = detector.stitcher.stitch_ref((obj_img, ref_img), cfg_sys)
                    Hmat = tmp_img_dict['homomtx']
                    dx, dy = check_relative_pos_side_back(Hmat, (h, w))
                    print('final idx, n_ref, dx, dy:', idx, n_ref, dx, dy)
                    final_pair[idx, 2] = n_ref
                    delta = delta + n_ref - n_ref_0
                    print("the delta is", delta)
                    final_pair[idx, 3] = delta
                    find_pair = True
                    if count == 1:
                        np.savetxt("start_obj_offsite.txt", np.array([idx, delta]), fmt='%d')
                    print("Neighbor for object {} is found to be the reference {}".format(idx, n_ref))
                    cv2.destroyAllWindows()
                elif kb == ord("p"):
                    find_pair = True
                    cv2.destroyAllWindows()
                else:
                    print("WRONG KEY")
                    print("press b for back one ref, n for next ref, s to select ref")
                    print("press p for passing to the next obj")
                    cv2.waitKey(0)
                time.sleep(0.1)
    df = pd.DataFrame(data=final_pair, columns=['obj_id', 'ref_id_start', 'ref_id_final', 'delta'])
    df.to_csv('neighbors_manually.csv')



def offline_process_GPS(cfg_path: AnyStr, start_obj, delta, mask):
    """offline process

    Args:
        cfg_path:
        ref_img_dir:
        curr_img_dir:

    Returns:

    """
    cfg_sys = fpp.ini_to_dict(cfg_path)
    detector = AirportRunwayFFCmp(**cfg_sys["Detector"])
    curr_paths = generate_folder_path(cfg_sys["General"]["curr_dir"])
    print('the current path', curr_paths['image'])
    curr_raw_gps = load_gps(curr_paths["sensor_data"])
    ref_paths = generate_folder_path(cfg_sys["General"]["ref_dir"])
    ref_raw_gps = load_gps(ref_paths["sensor_data"])
    curr_img_paths = glob.glob(os.path.join(str(curr_paths['image']), cfg_sys['Camera']['serial'], "*" + IMG_EXT))
    ref_img_paths = glob.glob(os.path.join(str(ref_paths['image']), cfg_sys['Camera']['serial'], "*" + IMG_EXT))
    curr_img_paths.sort()
    ref_img_paths.sort()
    dy_sign = find_dy_sign(detector, cfg_sys, ref_img_paths, nref=2)
    number_curr_img = len(curr_img_paths)
    ref_ts = [float(os.path.splitext(os.path.split(x)[-1])[0]) for x in ref_img_paths]
    ref_ts = np.asarray(ref_ts).reshape(-1, 1)  # become one column nx1 shape

    ref_img_gps = generate_metadata(ref_ts, ref_raw_gps)
    ref_img_gps = pd.DataFrame(ref_img_gps, columns=["ts", "lat", "lon"])
    curr_gps = {}
    delta_lat = 0
    delta_lon = 0
    neighbor_info = []
    table = np.zeros((number_curr_img, 12))
    for id_obj, curr_img_path in enumerate(curr_img_paths):
        if id_obj >= start_obj:  # the first object img may have no reference
            curr_img = load_img(curr_img_path, mask=mask)
            curr_ts = float(os.path.splitext(os.path.split(curr_img_path)[-1])[0])
            curr_ts = np.asarray(curr_ts).reshape(-1, 1)
            curr_img_gps = generate_metadata(curr_ts, curr_raw_gps)
            curr_gps["lat"] = curr_img_gps[0, 1] + delta_lat
            curr_gps["lon"] = curr_img_gps[0, 2] + delta_lon
            neighbor_name = find_best_gps_neighbor(curr_gps, ref_img_gps,
                                                   dist_threshold=int(cfg_sys["General"]["preload_radius"]))
            # neighbor_name = float(neighbor_1st[id_obj,1])
            id_ref = np.argmax(ref_ts == neighbor_name)
            table[id_obj, 0] = id_obj
            table[id_obj, 1] = id_ref
            table[id_obj, 2] = delta

            neighbor_info.append([curr_ts, neighbor_name, id_obj, id_ref])
            print("id_ref before correction", id_obj, id_ref)
            id_ref = id_ref + delta  # take care of gps correction
            id_ref_0 = id_ref
            print("id_ref after correction", id_obj, id_ref)
            if neighbor_name is None:
                detector.logger.info("No neighbor is found for {}".format(curr_img_path))
                continue
            # print('the reference_img_path is \n', os.path.join(ref_img_dir, str(neighbor_name) + IMG_EXT) )
            ref_img = load_img(ref_img_paths[id_ref], mask=mask)
            #
            homomat = detector.stitcher.stitch_homo((curr_img, ref_img), cfg_sys)
            # tmp_img_dict = detector.stitcher.stitch_ref((curr_img, ref_img), cfg_sys)
            if homomat is None:
                detector.logger.info("No neighbor is found for {}".format(curr_img_path))
                continue
            else:
                tic = time.time()
                table, homomat, best_ref = find_best_neighbor(detector, homomat, curr_img, id_obj, id_ref,
                                                              ref_img_paths, table, cfg_sys, dy_sign, mask)
                tac = time.time()
                table[id_obj, 10] = tac - tic
            if homomat is not None:
                ref_img = load_img(ref_img_paths[best_ref], mask=mask)
                tic = time.time()
                img_dict = detector.stitcher.stitch_wrap((curr_img, ref_img), homomat)
                tac = time.time()
                print('the wrap time', tac-tic)
                id_ref = best_ref
                #delta = delta + id_ref - id_ref_0
                delta_lat = delta_lat + ref_img_gps.loc[id_ref]['lat'] - ref_img_gps.loc[id_ref_0]['lat']
                delta_lon = delta_lon + ref_img_gps.loc[id_ref]['lon'] - ref_img_gps.loc[id_ref_0]['lon']
                tac = time.time()
                table[id_obj, 11] = tac - tic
            else:
                detector.logger.info("No neighbor is found for {}".format(curr_img_path))
                continue
            if img_dict is None:
                detector.logger.info("No neighbor is found for {}".format(curr_img_path))
                continue
            else:
                stitch_cmp(detector, img_dict, curr_img, id_obj, id_ref, cfg_sys)

    df = pd.DataFrame(data=table,
                      columns=['obj', 'ref_1', 'delta', 'ref_2', 'dx', 'dy', 'ref_new', 'dx_final', 'dy_final',
                               'thresh_y', 'time(s)', 'time_cpp'])
    df.to_csv(os.path.join(detector.output_path, "check_dy_neighbor" + ".csv"))
    neighbor_arr = np.asarray(neighbor_info).reshape(-1, 4)
    np.savetxt(os.path.join(detector.output_path, "neighbor_info.txt"), neighbor_arr)
    log_file = '/home/flyinstinct/Documents/pycharm_logs/log'
    os.system('mv ' + log_file + ' ' + detector.output_path)
    #    print("the output path is", detector.output_path)
    #    print('the split of detector out put path', detector.output_path.split('/')[-1])
    if SAVE_GOOGLE_DRIVE:
        my_drive_dir = '/home/flyinstinct/Desktop/google_drive'
        # here to mount google-drive with my local dir
        os.system('google-drive-ocamlfuse' + ' ' + my_drive_dir)
        # the directory where you want to rsync to google drive
        source_dir = detector.output_path
        # the directory where you linked to the google_drive, I recommend you to
        # keep the tree of the output so you do not mess up the run from different
        # days.
        target_dir = my_drive_dir + '/test_rsync/' + detector.output_path.split('/')[-1]
        os.system('rsync --update -zavh' + ' ' + source_dir + ' ' + target_dir)
        # unmount after rsync if you want
        os.system('fusermount -u' + ' ' + my_drive_dir)


def main():
    start_obj = 0
    offsite = 0
    cfg_file = "config_19195148.ini"  ## modify it when changing reference and observation datasets
    mask = np.zeros((3000, 4000)).astype(np.uint8)
    mask[:, :-1000] = 1
    tic = time.time()
    offline_process_jry(cfg_file, start_obj, offsite, mask)

    tac = time.time()
    print("the total running time is: ", tac - tic)

if __name__ == '__main__':
    main()
