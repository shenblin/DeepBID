import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import os

@DATASET_REGISTRY.register()
class train3DDataset(data.Dataset):
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
        super(train3DDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.gt_folder = sorted(os.listdir(self.gt_root))
        self.lq_folder = sorted(os.listdir(self.lq_root))

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # get GT
        img_gts = []
        im_path = os.path.join(self.gt_root, self.gt_folder[index])
        im_files = sorted(os.listdir(im_path))
        for k, im_file0 in enumerate(im_files):
            img_gt_path = os.path.join(im_path, im_file0)
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        img_lqs = []
        im_path = os.path.join(self.lq_root, self.lq_folder[index])
        im_files = sorted(os.listdir(im_path))
        for k, im_file0 in enumerate(im_files):
            img_lq_path = os.path.join(im_path, im_file0)
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

        interval = len(img_gts) // len(img_lqs)  # Temporal interval between input and GT

        # randomly crop
        try:
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, self.opt['crop_size'], self.opt['scale'])
        except:
            pass

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // (1+interval):], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // (1+interval)], dim=0)

        if self.opt['gray']:
            img_gts = img_gts[:,:1,:,:]  # gray
            img_lqs = img_lqs[:,:1,:,:]

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': self.gt_folder}

    def __len__(self):
        return len(self.gt_folder)
