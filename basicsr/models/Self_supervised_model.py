import torch
from collections import Counter
from torch import distributed as dist
from tqdm import tqdm
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel
import os
import time
import numpy as np
from skimage import io


@MODEL_REGISTRY.register()
class Self_supervised_RecurrentModel(VideoBaseModel):

    def __init__(self, opt):
        super(Self_supervised_RecurrentModel, self).__init__(opt)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None
        stitching = self.opt['val']['stitching']

        rank, world_size = get_dist_info()

        metric_data = dict()
        num_stacks = len(dataset)
        num_pad = (world_size - (num_stacks % world_size)) % world_size
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='stack')
        # Will evaluate (num_stacks + num_pad) times, but only the first num_stacks results will be recorded.
        # (To avoid wait-dead)
        for i in range(rank, num_stacks + num_pad, world_size):
            idx = min(i, num_stacks - 1)
            val_data = dataset[idx]
            patch = val_data['patch']
            im_mean = val_data['im_mean']
            # print(im_mean)

            # compute outputs
            val_data['lq'].unsqueeze_(0)
            val_data['gt'].unsqueeze_(0)
            self.feed_data(val_data)
            val_data['lq'].squeeze_(0)
            val_data['gt'].squeeze_(0)

            # t1 = time.time()
            self.test()
            # t2 = time.time()
            # print('image sequence: %d, inference time:%.2f' % (idx, (t2-t1)))
            visuals = self.get_current_visuals()

            # tentative for out of GPU memory
            del self.lq
            del self.output
            if 'gt' in visuals:
                del self.gt
            torch.cuda.empty_cache()

            # evaluate
            if i < num_stacks:
                # print(visuals['result'].shape)
                output_patch = np.squeeze(visuals['result'].cpu().detach().numpy())
                output_patch += im_mean
                raw_patch = np.squeeze(visuals['lq'].cpu().detach().numpy())
                raw_patch += im_mean

                output_patch = output_patch * (np.sum(raw_patch) / np.sum(output_patch)) ** 0.5
                output_patch = output_patch.squeeze().astype(np.float32)
                output_patch = output_patch.astype('int8')

                raw_patch = raw_patch.squeeze().astype(np.float32)
                raw_patch = raw_patch.astype('int8')

                if save_img:
                    if self.opt['is_train']:
                        img_path = os.path.join(self.opt['path']['visualization'], str(current_iter))
                    else:
                        img_path = os.path.join(self.opt['path']['visualization'], current_iter)
                    if not os.path.exists(img_path):
                        os.makedirs(img_path)
                    io.imsave(os.path.join(img_path, patch), output_patch, check_contrast=False)

                # calculate metrics
                    if with_metrics:
                        if not hasattr(self, 'metric_results'):  # only execute in the first run
                            self.metric_results = {}
                            # num_patch_each_stack = Counter(dataset.data_info['patch'])
                            for _, patch0 in enumerate(dataset.data_info['patch']):
                                self.metric_results[patch0] = torch.zeros(
                                    output_patch.shape[0], len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
                        # initialize the best metric results
                            self._initialize_best_metric_results(dataset_name)

                        metric_data['img2'] = np.mean(raw_patch, axis=0)
                        for i_frame in range(output_patch.shape[0]):
                            metric_data['img'] = output_patch[i_frame,:,:]
                            for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                                result = calculate_metric(metric_data, opt_)
                                self.metric_results[patch][i_frame, metric_idx] += result
                # progress bar
                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'stack: {patch}')

        if rank == 0:
            pbar.close()

        if with_metrics:
            if self.opt['dist']:
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                dist.barrier()

            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        if stitching:
            if save_img:
                overlap = dataset.opt['overlap']
                im_names = sorted(os.listdir(img_path))
                k_stitch = 0
                k_t_new = 0
                for k, im_name in enumerate(im_names):
                    val_data = dataset[k]
                    h_num, w_num, t_num = val_data['split_num'], val_data['split_num'], val_data['split_t_num']
                    im = io.imread(os.path.join(img_path, im_name))
                    i = k_stitch // w_num  # i row idx
                    j = k_stitch % w_num  # j col idx

                    # t, h, w = im.shape
                    if i == 0:
                        im = im[:, :-2 * overlap, :]
                    elif i == h_num - 1:
                        im = im[:, 2 * overlap:, :]
                    else:
                        im = im[:, overlap:-overlap, :]
                    if j == 0:
                        im = im[:, :, :-2 * overlap]
                    elif j == w_num - 1:
                        im = im[:, :, 2 * overlap:]
                    else:
                        im = im[:, :, overlap:-overlap]
                    t, h, w = im.shape
                    if k_stitch % (h_num * w_num) == 0:
                        h_s, w_s = h, w
                        h_t, w_t = h_num * h_s, w_num * w_s
                        im_stitch = np.zeros([t, h_t, w_t], dtype=np.uint8)

                    assert h == h_s, w == w_s
                    # im_t[h_s * i:h_s * (i + 1), w_s * j:w_s * (j +a 1)] = im

                    im_stitch[:, h_s * i:h_s * (i + 1), w_s * j:w_s * (j + 1)] = im
                    os.remove(os.path.join(img_path, im_name))

                    if (k + 1) % (h_num * w_num) == 0:

                        if k // (h_num * w_num) == 0 or k_t_new == 1:
                            im_sequence = im_stitch
                        else:
                            im_sequence = np.concatenate((im_sequence, im_stitch), axis=0)
                        k_t_new = 0
                        k_stitch = 0
                    else:
                        k_stitch += 1

                    if (k + 1) % (t_num * h_num * w_num) == 0:
                        im_stitch_name = ''
                        for _, im_stitch_name0 in enumerate(im_name.split('_')[:-2]):
                            im_stitch_name += '_'+ im_stitch_name0
                        print('stitching  ', im_stitch_name)
                        if self.opt['is_train']:
                            stitch_path = os.path.join(img_path,img_path.split(str(current_iter))[0],
                                                   img_path.split('/')[-1] + im_stitch_name + '.tif')
                        else:
                            stitch_path = os.path.join(img_path.split('visualization')[0],
                                                   img_path.split('/')[-1] + im_stitch_name + '.tif')
                        io.imsave(stitch_path, im_sequence, check_contrast=False)
                        del im_sequence
                        k_t_new = 1


    def test(self):
        n = self.lq.size(1)
        self.net_g.eval()

        flip_seq = self.opt['val'].get('flip_seq', False)

        if flip_seq:
            self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)

        with torch.no_grad():
            self.output = self.net_g(self.lq)

        if flip_seq:
            output_1 = self.output[:, :n, :, :, :]
            output_2 = self.output[:, n:, :, :, :].flip(1)
            self.output = 0.5 * (output_1 + output_2)

        self.net_g.train()
