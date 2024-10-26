import config as c
from train import *
import torch
import os
from dataset import MVTecDataset, Btad_Dataset
import random
import numpy as np
from utils import *

os.environ['TORCH_HOME'] = r'models/EfficientNet'
# Mvtec
class_name = ['cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
              'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 'metal_nut', 'zipper']

# btad
# class_name = ['01', '02', '03']

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    init_seeds(9826)
    det_list = list()
    loc_list = list()
    for i in class_name:
        c.class_name = i
        train_dataset = MVTecDataset(c, is_train=True)
        test_dataset = MVTecDataset(c, is_train=False)

        # train_dataset = Btad_Dataset(c, is_train=True)
        # test_dataset = Btad_Dataset(c, is_train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True,
                                                   num_workers=c.workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False,
                                                  num_workers=c.workers, pin_memory=True)

        res = train_loc(train_loader, test_loader)
        det_list.append(res[0])
        loc_list.append(res[1])
    print("Det.AUROC{}".format(sum(det_list) / len(det_list)))
    print("loc.AUROC{}".format(sum(loc_list) / len(loc_list)))
