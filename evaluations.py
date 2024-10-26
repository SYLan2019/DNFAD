
from functools import partial
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from skimage.measure import label, regionprops
import config as c
import numpy as np
import cv2
import torch
import os
from utils import rescale, t2np
from PIL import Image

loc_export_dir = './viz/loc'


def eval_det_auroc(det_auroc_obs, epoch, gt_label, anomaly_score):
    det_auroc = roc_auc_score(gt_label, anomaly_score) * 100
    best = det_auroc_obs.update(det_auroc, epoch)
    return det_auroc, best


def eval_loc_auroc(loc_auroc_obs, epoch, gt_mask, anomaly_score_map):
    loc_auroc = roc_auc_score(gt_mask.flatten(), anomaly_score_map.flatten()) * 100
    best = loc_auroc_obs.update(loc_auroc, epoch)
    return loc_auroc, best


def eval_seg_pro(loc_pro_obs, epoch, gt_mask, anomaly_score_map, max_step=800):
    expect_fpr = 0.3  # default 30%
    max_th = anomaly_score_map.max()
    min_th = anomaly_score_map.min()
    delta = (max_th - min_th) / max_step
    threds = np.arange(min_th, max_th, delta).tolist()

    pool = Pool(8)
    ret = pool.map(partial(single_process, anomaly_score_map, gt_mask), threds)
    pool.close()
    pros_mean = []
    fprs = []
    for pro_mean, fpr in ret:
        pros_mean.append(pro_mean)
        fprs.append(fpr)
    pros_mean = np.array(pros_mean)
    fprs = np.array(fprs)
    idx = fprs < expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    loc_pro_auc = auc(fprs_selected, pros_mean_selected) * 100
    best = loc_pro_obs.update(loc_pro_auc, epoch)

    return loc_pro_auc, best


def single_process(anomaly_score_map, gt_mask, thred):
    binary_score_maps = np.zeros_like(anomaly_score_map, dtype=np.bool_)
    binary_score_maps[anomaly_score_map <= thred] = 0
    binary_score_maps[anomaly_score_map > thred] = 1
    pro = []
    for binary_map, mask in zip(binary_score_maps, gt_mask):  # for i th image
        for region in regionprops(label(mask)):
            axes0_ids = region.coords[:, 0]
            axes1_ids = region.coords[:, 1]
            tp_pixels = binary_map[axes0_ids, axes1_ids].sum()
            pro.append(tp_pixels / region.area)

    pros_mean = np.array(pro).mean()
    inverse_masks = 1 - gt_mask
    fpr = np.logical_and(inverse_masks, binary_score_maps).sum() / inverse_masks.sum()
    return pros_mean, fpr


# viz
def min_max_norm_v2(image, max_value):
    a_min, a_max = image.min(), image.max()

    restricted = (image - a_min) / (max_value - a_min)
    return restricted


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap) / 255 + np.float32(image) / 255
    out = out / np.max(out)
    return np.uint8(255 * out)


def image_with_ground_truth(img, gt, kernel_size=(7, 7)):
    kernel = np.ones(kernel_size, np.uint8)
    gt = np.float32(gt)
    erode_gt = cv2.erode(gt, kernel)
    dilate_gt = cv2.dilate(gt, kernel)
    edge_gt = dilate_gt - erode_gt

    red_edge_gt = np.zeros(shape=(edge_gt.shape + (3,)))
    red_edge_gt[:, :, 1] = 128 * edge_gt
    red_edge_gt[:, :, 2] = 255 * edge_gt

    img_part_mask = np.zeros_like(red_edge_gt)
    img_part_mask[:, :, 0] = img_part_mask[:, :, 1] = img_part_mask[:, :, 2] = (edge_gt < 0.1)

    out = red_edge_gt + img * img_part_mask
    return np.uint8(out)


def image_with_predicted_mask(img, amap, threshold, kernel_size=(3, 3)):
    predicted_mask = np.float32(amap > threshold)
    kernel = np.ones(kernel_size, np.uint8)
    erode_mask = cv2.erode(predicted_mask, kernel)
    dilate_mask = cv2.dilate(predicted_mask, kernel)
    edge_mask = dilate_mask - erode_mask

    red_edge_mask = np.zeros(shape=(edge_mask.shape + (3,)))
    red_edge_mask[:, :, 1] = 128 * edge_mask
    red_edge_mask[:, :, 2] = 255 * edge_mask

    img_part_mask = np.zeros_like(red_edge_mask)
    img_part_mask[:, :, 0] = img_part_mask[:, :, 1] = img_part_mask[:, :, 2] = (edge_mask < 0.1)

    out = red_edge_mask + img * img_part_mask
    return np.uint8(out)


def heatmap_on_image_with_prediced_mask(heatmap_on_img, amap, threshold, kernel_size=(3, 3)):
    predicted_mask = np.float32(amap > threshold)
    kernel = np.ones(kernel_size, np.uint8)
    erode_mask = cv2.erode(predicted_mask, kernel)
    dilate_mask = cv2.dilate(predicted_mask, kernel)
    edge_mask = dilate_mask - erode_mask

    red_edge_mask = np.zeros(shape=(edge_mask.shape + (3,)))
    red_edge_mask[:, :, 1] = 128 * edge_mask
    red_edge_mask[:, :, 2] = 255 * edge_mask

    img_part_mask = np.zeros_like(red_edge_mask)
    img_part_mask[:, :, 0] = img_part_mask[:, :, 1] = img_part_mask[:, :, 2] = (edge_mask < 0.1)

    out = red_edge_mask + heatmap_on_img * img_part_mask
    return np.uint8(out)


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, gt_mask_list,
                 path_list,
                 anomaly_score_map_add, anomaly_score_map_mul, pro_eval):
    #
    gt_label = np.asarray(gt_label_list, dtype=np.bool_)
    gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=np.bool_), axis=1)
    det_auroc, best_det_auroc = eval_det_auroc(det_auroc_obs, epoch, gt_label, anomaly_score)

    loc_auroc, best_loc_auroc = eval_loc_auroc(loc_auroc_obs, epoch, gt_mask, anomaly_score_map_add)
    if best_loc_auroc and c.viz:
        # f1-score thresh-hold pixel
        precision, recall, threshholds = precision_recall_curve(gt_mask.ravel(),
                                                                anomaly_score_map_add.ravel())
        f1_scores = 2 * recall * precision / (recall + precision + 1e-6)
        best_threshold = threshholds[np.argmax(f1_scores)]
        # best_threshold = 0.55
        print('Best threshold:', best_threshold)

        maximum = np.max(anomaly_score_map_add)
        for i, gt in enumerate(gt_mask):

            img = np.array(Image.open(path_list[i]))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (gt.shape[0], gt.shape[1]))
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # save
            root = loc_export_dir
            img_path = os.path.join(root, c.class_name, str(i), f"{i}.png")
            gt_path = os.path.join(root, c.class_name, str(i), f"{i}_mask.png")
            amap_path = os.path.join(root, c.class_name, str(i), f"{i}_amap.png")

            os.makedirs(os.path.join(root, c.class_name, str(i)), exist_ok=True)

            amap_on_img_path = os.path.join(root, c.class_name, str(i), f"{i}_amap_on_img.png")
            img_with_gt_path = os.path.join(root, c.class_name, str(i), f"{i}_img_with_gt.png")
            img_with_mask_path = os.path.join(root, c.class_name, str(i), f"{i}_img_with_mask.png")
            heatmap_on_img_with_mask_path = os.path.join(root, str(i), c.class_name,
                                                         f"{i}_heatmap_on_img_with_mask.png")

            amap = anomaly_score_map_add[i]
            heatmap = cvt2heatmap(min_max_norm_v2(amap, max_value=maximum) * 255)
            heatmap_on_img = heatmap_on_image(heatmap, img)
            #
            img_with_gt = image_with_ground_truth(img, gt)
            img_with_mask = image_with_predicted_mask(img, amap, best_threshold)

            heatmap_on_img_with_mask = heatmap_on_image_with_prediced_mask(heatmap_on_img, amap, best_threshold)

            cv2.imwrite(img_path, img)
            cv2.imwrite(gt_path, gt * 255)
            cv2.imwrite(amap_path, heatmap)

            cv2.imwrite(amap_on_img_path, heatmap_on_img)
            cv2.imwrite(img_with_gt_path, img_with_gt)
            cv2.imwrite(img_with_mask_path, img_with_mask)
            cv2.imwrite(heatmap_on_img_with_mask_path, heatmap_on_img_with_mask)

    if pro_eval:
        seg_pro_auc, best_loc_pro = eval_seg_pro(loc_pro_obs, epoch, gt_mask, anomaly_score_map_mul)
    else:
        seg_pro_auc, best_loc_pro = None, False


    return best_det_auroc, best_loc_auroc, best_loc_pro
