import os
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

import config as c
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import numpy as np
from glob import glob
from PIL import Image
from dataset import *
import datetime

_GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))
log_theta = torch.nn.LogSigmoid()

RESULT_DIR = './result'
def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def get_loss(z, jac):
    '''check equation 4 of the paper why this makes sense - oh and just ignore the scaling here'''
    z = z.reshape(z.shape[0], -1)
    return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - jac) / z.shape[1]

def get_logp(C, z, logdet_J):
    logp = C * _GCONST_ - 0.5*torch.sum(z**2, 1) + logdet_J
    # logp = - 0.5 * torch.sum(z ** 2, 1)
    return logp

def cat_maps(z):
    return torch.cat([z[i].reshape(z[i].shape[0], -1) for i in range(len(z))], dim=1)


class Score_Observer:
    def __init__(self, name, total_epochs):
        self.name = name
        self.total_epochs = total_epochs
        self.max_epoch = 0
        self.max_score = 0.0
        self.last_score = 0.0

    def update(self, score, epoch, print_score=True):
        self.last = score
        best = False
        if score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
            best = True
        if print_score:
            self.print_score(epoch)

        return best

    def print_score(self, epoch):
        print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
              'Epoch [{:d}/{:d}] {:s}: last: {:.2f}\tmax: {:.2f}\tepoch_max: {:d}'.format(
                  epoch, self.total_epochs - 1, self.name, self.last, self.max_score, self.max_epoch))




