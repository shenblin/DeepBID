import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

import os, skimage


@DATASET_REGISTRY.register()
class stack3DDataset(data.Dataset):
    """3D dataset for training recurrent networks.

    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).
    """

    def __init__(self, opt):
        super(stack3DDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.img_gt_name = sorted(os.listdir(self.gt_root))
        self.img_lq_name = sorted(os.listdir(self.lq_root))

        # temporal sparse interval
        self.sparse_interval = opt.get('sparse_interval', 1)

    def __getitem__(self, index):

        interval = self.sparse_interval

        # get the LQ and GT frames
        img_lqs = []
        img_gts = []

        # get GT
        img_gt_path = self.gt_root / self.img_gt_name[index]
        img_gt = skimage.io.imread(img_gt_path)
        img_gt = skimage.img_as_float(img_gt).astype(np.float32)  # uint8 to float32
        c_gt, h_gt, w_gt = img_gt.shape[0:3]
        for c in range(c_gt):
            img_gts.append(np.expand_dims(img_gt[c, :, :], axis=-1))

        # get LQ
        img_lq_path = self.lq_root / self.img_lq_name[index]
        img_lq = skimage.io.imread(img_lq_path)
        img_lq = skimage.img_as_float(img_lq).astype(np.float32)  # uint8 to float32
        c_lq, h_lq, w_lq = img_lq.shape[0:3]
        for c in range(c_lq):
            img_lqs.append(np.expand_dims(img_lq[c, :, :], axis=-1))

        # randomly crop
        try:
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, self.opt['crop_size'], self.opt['scale'])
        except:
            pass

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // (1 + interval):], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // (1 + interval)], dim=0)
        # print(img_gts.shape)
        # print(img_lqs.shape)
        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': self.img_gt_name}

    def __len__(self):
        return len(self.img_gt_name)
