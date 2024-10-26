import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import config as c
from model import ADwithGlow, save_model, save_weights, load_weights
from utils import *
import datetime
from postprocess import post_process
from evaluations import *
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



def model_forward(model, image):
    if c.extractor == 'dino':
        feature_map_list = model.dino_ext(image)
    elif c.extractor == 'wide_resnet50_2':
        feature_map_list = model.feature_extractor(image)
    elif c.extractor == 'efficient':
        feature_map_list = [model.feature_extractor(image)]
    pool_layer = nn.AvgPool2d(3, 2, 1)

    # feature_map_size = tuple(feature_map_list[1].shape[-2:])
    if c.extractor == 'wide_resnet50_2' or c.extractor == 'resnet18':
        feature_map_size = tuple(feature_map_list[2].shape[-2:])
        feature_map = [F.interpolate(x, size=feature_map_size, mode='bilinear', align_corners=False)
                       for i, x in enumerate(feature_map_list)]
        feature_map = torch.cat(feature_map, dim=1)
    if c.extractor == 'dino':
        feature_map = feature_map_list[0]

    return model(feature_map, feature_map)


def eval_loc(test_loader):
    model = ADwithGlow()
    model.feature_extractor.eval()
    model = load_weights(model, c.class_name + ".pth")

    print(F'\ntest on {c.class_name}')

    # evaluate
    model.map_flow.eval()
    model.patch_flow.eval()
    if c.verbose:
        print('\nCompute loss and scores on test set:')
    test_loss = 0.
    imag_count = 0
    test_z = [list() for i in range(2)]
    test_labels = list()
    gt_mask = list()
    img_list = list()
    size_list = list()
    start = time.time()

    global_z = []
    local_z = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
            inputs, labels, mask, path = data
            test_labels.extend(t2np(labels))
            gt_mask.extend(t2np(mask))

            img_list.extend(path)
            inputs = inputs.to(c.device)

            z1, jac1, z2, jac2 = model_forward(model, inputs)

            global_z.append(z2.reshape(z2.shape[0], -1))
            local_z.append(z1.reshape(z1.shape[0], -1))

            if (i == 0):
                size_list.append(tuple(z1.shape[-2:]))
                size_list.append(tuple(z2.shape[-2:]))
            logp1 = -0.5 * torch.mean(z1 ** 2, 1)
            test_z[0].append(logp1)
            logp2 = - 0.5 * torch.mean(z2 ** 2, 1)
            loss = torch.mean((0.5 * torch.sum(z2 ** 2, (1, 2, 3)) - jac2)) + torch.mean(
                (0.5 * torch.sum(z1 ** 2, 1) - jac1))

            test_z[1].append(logp2)
            test_loss += t2np(loss)
            imag_count += inputs.shape[0]

        anomaly_score_map_patch_level, anomaly_score_map_map_level, anomaly_score_map_level = post_process(c, test_z)


        test_loss /= imag_count
        fps = len(test_loader.dataset) / (time.time() - start)
        if c.verbose:
            print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
                  'test loss: {:.3e}\tFPS: {:.1f}'.format(
                      test_loss, fps))


class_name = ['metal_nut','bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
              'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

if __name__ == '__main__':
    det_list = list()
    loc_list = list()
    for i in class_name:
        c.class_name = i

        test_dataset = MVTecDataset(c, is_train=False)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                  num_workers=c.workers, pin_memory=True)
        eval_loc(test_loader)

