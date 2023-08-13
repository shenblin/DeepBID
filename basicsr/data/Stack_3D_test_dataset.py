import os, skimage
import torch
import numpy as np
from torch.utils import data as data

from basicsr.data.data_util import generate_frame_indices
from basicsr.utils import get_root_logger, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


# cache data or save the frame list
def read_img_stack(path, img_name):
    imgs = []
    img = skimage.io.imread(os.path.join(path, img_name))
    img = skimage.img_as_float(img).astype(np.float32)  # 从uint8转换成float32
    c_lq, h_lq, w_lq = img.shape[0:3]
    for c in range(c_lq):
        imgs.append(np.expand_dims(img[c, :, :], axis=-1))

    imgs = img2tensor(imgs)
    imgs = torch.stack(imgs, dim=1)
    # print(imgs.shape)

    return imgs


@DATASET_REGISTRY.register()
class Stack_3D_Test_Dataset(data.Dataset):
    """Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            name (str): Dataset name.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(Stack_3D_Test_Dataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'stack': []}

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        self.imgs_lq, self.imgs_gt = {}, {}

        self.img_gt_name = sorted(os.listdir(self.gt_root))
        self.img_lq_name = sorted(os.listdir(self.lq_root))

        self.data_info['lq_path'].extend(self.lq_root)
        self.data_info['gt_path'].extend(self.gt_root )

        for k in range(len(self.img_gt_name)):
            img_stack_name = self.img_gt_name[k]
            img = skimage.io.imread(os.path.join(self.gt_root, img_stack_name))
            max_idx = img.shape[0]
            self.data_info['stack'].extend([img_stack_name] * max_idx)
            logger.info(f' {img_stack_name} for VideoTestDataset...')

        self.stacks = sorted(list(set(self.data_info['stack'])))

    def __getitem__(self, index):

        stack = self.stacks[index]

        imgs_lq = read_img_stack(self.lq_root, stack)
        imgs_gt = read_img_stack(self.gt_root, stack)

        return {
            'lq': imgs_lq,
            'gt': imgs_gt,
            'stack': stack,
        }

    def __len__(self):
        return len(self.stacks)
