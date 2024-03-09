import glob
import cv2
import numpy as np
import torch
from os import path as osp
from torch.utils import data as data

from basicsr.data.data_util import generate_frame_indices
from basicsr.utils import img2tensor, get_root_logger, scandir
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import mod_crop


def read_img_seq(path, require_mod_crop=False, scale=1, return_imgname=False, gray=False):
    """Read a sequence of images from a given folder path.

    Args:
        path (list[str] | str): List of image paths or image folder path.
        require_mod_crop (bool): Require mod crop for each image.
            Default: False.
        scale (int): Scale factor for mod_crop. Default: 1.
        return_imgname(bool): Whether return image names. Default False.

    Returns:
        Tensor: size (t, c, h, w), RGB, [0, 1].
        list[str]: Returned image name list.
    """
    if isinstance(path, list):
        img_paths = path
    else:
        img_paths = sorted(list(scandir(path, full_path=True)))
    imgs = [cv2.imread(v).astype(np.float32) / 255. for v in img_paths]

    if require_mod_crop:
        imgs = [mod_crop(img, scale) for img in imgs]

    imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
    imgs = torch.stack(imgs, dim=1)
    # print(imgs.shape)

    if gray:
        imgs = imgs[:1, :, :, :]  # gray

    if return_imgname:
        imgnames = [osp.splitext(osp.basename(path))[0] for path in img_paths]
        return imgs, imgnames
    else:
        return imgs

@DATASET_REGISTRY.register()
class SequenceTestDataset(data.Dataset):
    """Sequence test dataset.

    More generally, it supports testing dataset with following structures:

    dataroot
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(SequenceTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': []}
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        logger.info(f'Generate data info for SequenceTestDataset - {opt["name"]}')
        self.imgs_lq, self.imgs_gt = {}, {}
        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))


        for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
            # get frame list for lq and gt
            subfolder_name = osp.basename(subfolder_lq)
            print(subfolder_lq)
            img_paths_lq = sorted(list(scandir(subfolder_lq, full_path=True)))
            img_paths_gt = sorted(list(scandir(subfolder_gt, full_path=True)))

            max_idx = len(img_paths_gt)

            self.data_info['lq_path'].extend(img_paths_lq)
            self.data_info['gt_path'].extend(img_paths_gt)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append(f'{i}/{max_idx}')

            # cache data or save the frame list
            if self.cache_data:
                logger.info(f'Cache {subfolder_name} for SequenceTestDataset...')
                self.imgs_lq[subfolder_name] = read_img_seq(img_paths_lq, gray=opt['gray'])
                self.imgs_gt[subfolder_name] = read_img_seq(img_paths_gt, gray=opt['gray'])
            else:
                self.imgs_lq[subfolder_name] = img_paths_lq
                self.imgs_gt[subfolder_name] = img_paths_gt


    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)

        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
            img_gt = self.imgs_gt[folder][idx]
        else:
            img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
            imgs_lq = read_img_seq(img_paths_lq)
            img_gt = read_img_seq([self.imgs_gt[folder][idx]])
            img_gt.squeeze_(0)

        return {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': img_gt,  # (c, h, w)
            'folder': folder,  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/99
            'lq_path': lq_path  # center frame
        }

    def __len__(self):
        return len(self.data_info['gt_path'])


@DATASET_REGISTRY.register()
class SequenceRecurrentTestDataset(SequenceTestDataset):
    """Sequence test dataset for recurrent architectures, which takes LR Sequence
    frames as input and output corresponding HR Sequence frames.

    Args:
        Same as SequenceTestDataset.
        Unused opt:
            padding (str): Padding mode.

    """

    def __init__(self, opt):
        super(SequenceRecurrentTestDataset, self).__init__(opt)
        # Find unique folder strings
        self.folders = sorted(list(set(self.data_info['folder'])))

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder]
            imgs_gt = self.imgs_gt[folder]
        else:
            raise NotImplementedError('Without cache_data is not implemented.')

        return {
            'lq': imgs_lq,
            'gt': imgs_gt,
            'folder': folder,
        }

    def __len__(self):
        return len(self.folders)
