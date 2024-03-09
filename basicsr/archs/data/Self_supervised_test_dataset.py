import torch
import numpy as np
from torch.utils import data as data
from basicsr.utils import get_root_logger
from basicsr.utils.registry import DATASET_REGISTRY
import os, skimage
import tifffile as tiff


def read_img_stack(path, img_name):
    # img = skimage.io.imread(os.path.join(path, img_name))
    # img = skimage.img_as_float(img).astype(np.float32)  # 从uint8转换成float32
    noise_im = tiff.imread(os.path.join(path, img_name))
    img = noise_im.astype(np.float32)

    return img


@DATASET_REGISTRY.register()
class Self_supervised_Test_Dataset(data.Dataset):
    """Args:
        opt (dict): Config for test dataset. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            crop_size (int): cropping images into spatial patches for reducing memory requirement, preferably divisible by image pixels
            overlap: overlap between patches
    """

    def __init__(self, opt):
        super(Self_supervised_Test_Dataset, self).__init__()
        self.opt = opt
        self.lq_root = opt['dataroot_lq']

        logger = get_root_logger()
        logger.info(f'Generate data info for TestDataset - {opt["name"]}')
        self.data_info = {'lq_path': [], 'gt_path': [], 'patch': []}
        self.data_info['lq_path'].extend(self.lq_root)
        self.data_info['gt_path'].extend(self.lq_root)

        self.imgs_lq, self.imgs_gt, self.im_mean, self.split_num, self.split_t_num = {}, {}, {}, {}, {}

        self.img_lq_name = sorted(os.listdir(self.lq_root))

        for k in range(len(self.img_lq_name)):
            img_stack_name = self.img_lq_name[k]
            logger.info(f' {img_stack_name} for Testing ...')

            img = read_img_stack(self.lq_root, img_stack_name)
            im = img - img.mean()

            patch_pixel = self.opt['crop_size']
            overlap = self.opt['overlap']
            t_len = self.opt['t_crop_size']

            t, h, w = im.shape
            split_num = h // patch_pixel
            split_t_num = max(t // t_len, 1)
            logger.info(f' Shape: {im.shape},  pixel split num: {split_num},  t split num: {split_t_num}')

            for i in range(split_num):
                for j in range(split_num):
                    num = split_num * i + j
                    if i == 0:
                        h_start = patch_pixel * i
                    elif i == split_num - 1:
                        h_start = patch_pixel * i - 2 * overlap
                    else:
                        h_start = patch_pixel * i - overlap
                    if j == 0:
                        w_start = patch_pixel * j
                    elif j == split_num - 1:
                        w_start = patch_pixel * j - 2 * overlap
                    else:
                        w_start = patch_pixel * j - overlap
                    if i == split_num - 1:
                        h_end = patch_pixel * (i + 1)
                    elif i == 0:
                        h_end = patch_pixel * (i + 1) + 2 * overlap
                    else:
                        h_end = patch_pixel * (i + 1) + overlap
                    if j == split_num - 1:
                        w_end = patch_pixel * (j + 1)
                    elif j == 0:
                        w_end = patch_pixel * (j + 1) + 2 * overlap
                    else:
                        w_end = patch_pixel * (j + 1) + overlap

                    for tt in range(split_t_num):
                        image_patch = im[tt * t_len: (tt + 1) * t_len, h_start:h_end, w_start:w_end]
                        patch_name = img_stack_name.split('.')[0] + '_' + '%03d' % tt +'_' +'%05d' % num + '.tif'

                        self.data_info['patch'].extend([patch_name])

                        self.split_num[patch_name] = split_num
                        self.split_t_num[patch_name] = split_t_num
                        self.im_mean[patch_name] = img.mean()

                        self.imgs_lq[patch_name] = torch.from_numpy(np.expand_dims(image_patch, 0))
                        self.imgs_gt[patch_name] = torch.from_numpy(np.expand_dims(image_patch, 0))

        self.patches = sorted(list(set(self.data_info['patch'])))


    def __getitem__(self, index):

        patch = self.patches[index]
        im_mean = self.im_mean[patch]
        split_num = self.split_num[patch]
        split_t_num = self.split_t_num[patch]
        imgs_lq = self.imgs_lq[patch]
        imgs_gt = self.imgs_gt[patch]

        return {
            'lq': imgs_lq,
            'gt': imgs_gt,
            'patch': patch,
            'im_mean': im_mean,
            'split_num': split_num,
            'split_t_num': split_t_num,
        }

    def __len__(self):
        return len(self.patches)




