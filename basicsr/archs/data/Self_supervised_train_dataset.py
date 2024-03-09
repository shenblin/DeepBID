import torch
from torch.utils import data as data
from basicsr.data.transforms import random_transform
from basicsr.utils import get_root_logger
from basicsr.utils.registry import DATASET_REGISTRY

import numpy as np
import os, math, random, skimage
import tifffile as tiff


@DATASET_REGISTRY.register()
class Self_supervised_train_Dataset(data.Dataset):
    """
    Modified from Real-time denoising enables high-sensitivity fluorescence time-lapse imaging beyond the shot-noise limit.
                  Nat Biotechnol 41, 282–292 (2023). https://doi.org/10.1038/s41587-022-01450-8
    The original noisy stack is partitioned into thousands of 3D sub-stacks (patch) with the setting
    overlap factor in each dimension.

    Important Fields:
       opt: self_supervised_mode: frame  # line
       opt: patch_x, y, t = crop patch size
       overlap_factor: overlap factor for patches
       gap_x, y, t: patch gap.
       self.name_list : the coordinates of 3D patch are indexed by the patch name in name_list.
       self.coordinate_list : record the coordinate of 3D patch preparing for partition in whole stack.
       self.stack_index : the index of the noisy stacks.
       self.img_input_all : the collection of all noisy stacks.
    """
    def __init__(self, opt):
        self.opt = opt

        self.self_supervised_mode = self.opt['self_supervised_mode']
        self.datasets_path = opt['dataroot_lq']

        logger = get_root_logger()
        logger.info(f'Generate data info for TrainDataset - {opt["name"]}')
        self.patch_t = self.opt['patch_t']
        self.patch_x = self.opt['patch_x']
        try:
            self.patch_y = self.opt['patch_y']
        except:
            self.patch_y = self.patch_x
        self.overlap_factor = self.opt['overlap_factor']

        self.gap_x = int(self.patch_x * (1 - self.overlap_factor))  # patch gap in x
        self.gap_y = int(self.patch_y * (1 - self.overlap_factor))  # patch gap in y
        # self.gap_t = int(self.patch_t * (1 - self.overlap_factor))  # patch gap in t

        self.name_list = []
        self.coordinate_list = {}
        self.stack_index = []
        self.img_input_all = []

        self.stack_num = len(list(os.walk(self.datasets_path, topdown=False))[-1][-1])
        print('Total stack number -----> ', self.stack_num)

        ind = 0
        for im_name in list(os.walk(self.datasets_path, topdown=False))[-1][-1]:

            im_dir = self.datasets_path + '//' + im_name
            img_input = tiff.imread(im_dir)

            self.whole_x = img_input.shape[2]
            self.whole_y = img_input.shape[1]
            self.whole_t = img_input.shape[0]
            logger.info(f' {im_name}, :  {img_input.shape} for Training ...')

            # Calculate real gap_t
            w_num = math.floor((self.whole_x - self.patch_x) / self.gap_x) + 1
            h_num = math.floor((self.whole_y - self.patch_y) / self.gap_y) + 1
            s_num = math.ceil(self.opt['train_datasets_size'] / w_num / h_num / self.stack_num)
            self.gap_t = math.floor((self.whole_t - self.patch_t * 2) / (s_num - 1))
            # print(self.gap_t)

            # Minus mean before training
            img_input = img_input.astype(np.float32)
            img_input = img_input-img_input.mean()

            self.img_input_all.append(img_input)
            patch_t2 = self.patch_t * 2
            # print('int((whole_y-patch_y+gap_y)/gap_y) -----> ',int((self.whole_y - self.patch_y + self.gap_y) / self.gap_y))
            # print('int((whole_t-patch_t2+gap_t2)/gap_t2) -----> ',int((self.whole_t - patch_t2 + self.gap_t) / self.gap_t))
            for x in range(0, int((self.whole_y - self.patch_y + self.gap_y) / self.gap_y)):
                for y in range(0, int((self.whole_x - self.patch_x + self.gap_x) / self.gap_x)):
                    for z in range(0, int((self.whole_t - patch_t2 + self.gap_t) / self.gap_t)):
                        single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
                        init_h = self.gap_y * x
                        end_h = self.gap_y * x + self.patch_y
                        init_w = self.gap_x * y
                        end_w = self.gap_x * y + self.patch_x
                        init_s = self.gap_t * z
                        end_s = self.gap_t * z + patch_t2
                        # print(init_s, patch_t2, end_s)
                        single_coordinate['init_h'] = init_h
                        single_coordinate['end_h'] = end_h
                        single_coordinate['init_w'] = init_w
                        single_coordinate['end_w'] = end_w
                        single_coordinate['init_s'] = init_s
                        single_coordinate['end_s'] = end_s
                        patch_name = im_name.replace('.tif', '') + '_x' + str(x) + '_y' + str(y) + '_z' + str(z)
                        self.name_list.append(patch_name)
                        self.coordinate_list[patch_name] = single_coordinate
                        self.stack_index.append(ind)
            ind = ind + 1

    def __getitem__(self, index):
        """
        For temporal stacks with a small lateral size or short recording period, sub-stacks can be
        randomly cropped from the original stack to augment the training set according to the record
        coordinate. Then, interlaced frames of each sub-stack are extracted to form two 3D patches.
        One of them serves as the input and the other serves as the target for network training
        Args:
            index : the index of 3D patchs used for training
        Return:
            input, target : the consecutive frames of the 3D noisy patch serve as the input and target of the network
        """
        stack_index = self.stack_index[index]
        img_input = self.img_input_all[stack_index]
        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']

        if self.self_supervised_mode == 'frame':
            # print('frame')
            inputs = img_input[init_s:end_s:2, init_h:end_h, init_w:end_w]
            target = img_input[init_s + 1:end_s:2, init_h:end_h, init_w:end_w]
        elif self.self_supervised_mode == 'line':
            # print('line')
            inputs = np.concatenate((img_input[init_s:end_s:2, init_h:end_h:2, init_w:end_w],
                                    img_input[init_s + 1:end_s:2, init_h:end_h:2, init_w:end_w]), axis=1)
            target = np.concatenate((img_input[init_s:end_s:2, init_h + 1:end_h:2, init_w:end_w],
                                     img_input[init_s + 1:end_s:2, init_h + 1:end_h:2, init_w:end_w]), axis=1)
        else:
            raise IndexError('self_supervised_mode should be line or frame')
        # output_path = os.path.join(self.datasets_path, 'train_set')
        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)
        # print(self.name_list[index])
        # new_img_path = os.path.join(output_path, (self.name_list[index] + '.tif'))
        # skimage.io.imsave(new_img_path, inputs, check_contrast=False)
        # aa

        # generate a random number determinate whether swap input and target
        if random.random() < 0.5:
            temp = inputs
            inputs = target
            target = temp  # Swap inputs and target

        inputs, target = random_transform(inputs, target)

        inputs = torch.from_numpy(np.expand_dims(inputs, 0).copy())
        target = torch.from_numpy(np.expand_dims(target, 0).copy())

        # print(inputs.shape)
        return {'lq': inputs, 'gt': target, 'key': stack_index}

    def __len__(self):
        return len(self.name_list)
