import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import config as c
# from localization import export_gradient_maps
from model import ADwithGlow, save_model, save_weights
from utils import *
import datetime
from postprocess import post_process
from evaluations import *
import torch.nn.functional as F
import torch.nn as nn


def model_forward(model, image):
    if c.extractor == 'dino':
        feature_map_list = model.dino_ext(image)
    elif c.extractor == 'wide_resnet50_2':
        feature_map_list = model.feature_extractor(image)
    elif c.extractor == 'efficient':
        feature_map_list = [model.feature_extractor(image)]
    pool_layer = nn.AvgPool2d(3, 2, 1)

    if c.extractor == 'wide_resnet50_2' or c.extractor == 'resnet18':
        feature_map_size = tuple(feature_map_list[2].shape[-2:])
        feature_map = [F.interpolate(x, size=feature_map_size, mode='bilinear', align_corners=False)
                       for i, x in enumerate(feature_map_list)]
        feature_map = torch.cat(feature_map, dim=1)
    if c.extractor == 'dino':
        feature_map = feature_map_list[0]
    return model(feature_map, feature_map)


def train_loc(train_loader, test_loader):
    model = ADwithGlow()
    params = list(model.map_flow.parameters())
    # params = list(model.patch_flow.parameters())
    params += list(model.patch_flow.parameters())
    optimizer = torch.optim.AdamW(params, lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04,
                                  weight_decay=1e-5)

    # optim warm_up
    if c.lr_warmup:
        start_factor = c.lr_warmup_from
        end_factor = 1.0
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor,
                                                             end_factor=end_factor,
                                                             total_iters=
                                                             c.lr_warmup_epochs * c.sub_epochs)
    else:
        warmup_scheduler = None
    mile_stones = [milestone for milestone in c.lr_decay_milestones if milestone > 0]
    if mile_stones:
        decay_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, mile_stones, c.lr_decay_gamma)
    else:
        decay_scheduler = None

    model.to(c.device)

    det_auroc_obs = Score_Observer('Det.AUROC', c.meta_epochs)
    loc_auroc_obs = Score_Observer('Loc.AUROC', c.meta_epochs)
    loc_pro_obs = Score_Observer('Loc.PRO', c.meta_epochs)

    print(F'\nTrain on {c.class_name}')
    best_acc = 0
    for epoch in range(c.meta_epochs):
        model.map_flow.train()
        model.patch_flow.train()
        if c.verbose:
            print(F'\nTrain epoch {epoch}')
        for sub_epoch in range(c.sub_epochs):
            train_loss = list()
            for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                inputs, labels, _, path = data
                inputs = inputs.to(c.device)

                z1, jac1, z2, jac2 = model_forward(model, inputs)

                loss1 = torch.mean((0.5 * torch.sum(z1 ** 2, 1) - jac1))
                loss2 = (0.5 * torch.sum(z2 ** 2, (1, 2, 3)) - jac2)
                loss = loss1 + loss2.mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(params, 2)
                train_loss.append(t2np(loss))
                optimizer.step()
            mean_train_loss = np.mean(train_loss)
            # warm up
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            if warmup_scheduler:
                warmup_scheduler.step()
            if decay_scheduler:
                decay_scheduler.step()
            if c.verbose:
                print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
                      'Epoch {:d}.{:d} train loss: {:.3e}\tlr={:.2e}'.format(
                          epoch, sub_epoch, mean_train_loss, lr))
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

        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
                inputs, labels, mask, path = data
                test_labels.extend(t2np(labels))
                gt_mask.extend(t2np(mask))
                img_list.extend(path)
                inputs = inputs.to(c.device)
                z1, jac1, z2, jac2 = model_forward(model, inputs)

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

        test_loss /= imag_count
        fps = len(test_loader.dataset) / (time.time() - start)
        if c.verbose:
            print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
                  'Epoch {:d}   test loss: {:.3e}\tFPS: {:.1f}'.format(
                      epoch, test_loss, fps))
        # 指标计算
        anomaly_score, anomaly_score_map_loc, anomaly_score_map = post_process(c, test_z)
        best_det_auroc, best_loc_auroc, best_loc_pro = eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch,
                                                                    test_labels, anomaly_score, gt_mask, img_list,
                                                                    anomaly_score_map_loc, anomaly_score_map,
                                                                    False)

    return det_auroc_obs.max_score, loc_auroc_obs.max_score
