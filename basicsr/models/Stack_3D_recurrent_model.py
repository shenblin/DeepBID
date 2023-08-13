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
class Stack_3D_RecurrentModel(VideoBaseModel):

    def __init__(self, opt):
        super(Stack_3D_RecurrentModel, self).__init__(opt)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = self.net_g.parameters()
        optim_type = train_opt['optim_g'].pop('type')

        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        super(Stack_3D_RecurrentModel, self).optimize_parameters(current_iter)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {}
                num_frame_each_stack = Counter(dataset.data_info['stack'])
                for stack, num_frame in num_frame_each_stack.items():
                    self.metric_results[stack] = torch.zeros(
                        num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
            # initialize the best metric results
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        rank, world_size = get_dist_info()
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()

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
            stack = val_data['stack']

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
                num_frame, h, w = visuals['result'][0, 0, :, :, :].shape
                img_stack = np.zeros([num_frame, h, w], dtype=np.uint8)
                gt_stack = np.zeros([num_frame, h, w], dtype=np.uint8)

                for idx in range(visuals['result'].size(2)):
                    result = visuals['result'][0, :, idx, :, :]
                    result_img = tensor2img([result])  # uint8, bgr
                    if save_img:
                        img_stack[idx,:,:] = result_img
                    metric_data['img'] = result_img
                    if 'gt' in visuals:
                        gt = visuals['gt'][0, :, idx, :, :]
                        gt_img = tensor2img([gt])  # uint8, bgr
                        metric_data['img2'] = gt_img
                        gt_stack[idx, :, :] = gt_img

                    # calculate metrics
                    if with_metrics:
                        for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                            result = calculate_metric(metric_data, opt_)
                            self.metric_results[stack][idx, metric_idx] += result

                if save_img:
                    if self.opt['is_train']:
                        # raise NotImplementedError('saving image is not supported during training.')
                        img_path = os.path.join(self.opt['path']['visualization'], str(current_iter))
                        if not os.path.exists(img_path):
                            os.makedirs(img_path)
                        io.imsave(os.path.join(img_path, stack), img_stack, check_contrast=False)
                    else:
                        img_path = os.path.join(self.opt['path']['visualization'], current_iter)
                        if not os.path.exists(img_path):
                            os.makedirs(img_path)
                        io.imsave(os.path.join(img_path, stack), img_stack, check_contrast=False)
                # progress bar
                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'stack: {stack}')

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
