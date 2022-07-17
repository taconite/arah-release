import os
import cv2
import time
import numpy as np
from collections import defaultdict
from tqdm import tqdm


class BaseTrainer(object):
    ''' Base trainer class.
    '''

    def evaluate(self, val_loader, it, val_dataset=None, tf_logger=None, mesh_dir=None, image_dir=None, gen_cano_mesh=False):
        ''' Performs an evaluation.

        Args:
            val_loader (dataloader): pytorch dataloader for validation dataloader
            it (int): current training step
            val_dataset (torch.utils.data.Dataset): pytorch dataset for validation
            tf_logger (tensorboardX.SummaryWriter): TensorBoard writer
            mesh_dir (str): directory to save generated 3D meshes
            image_dir (str): directory to save rendered images
            gen (str): directory to save rendered images
            gen_cano_mesh (bool): whether to generate the canonical mesh or not
        '''
        eval_list = defaultdict(list)

        if hasattr(val_dataset, 'img_size'):
            img_height, img_width = val_dataset.img_size
        else:
            img_height = img_width = None

        d_idx = 0
        for data in tqdm(val_loader):
            t = time.time()
            eval_step_dict = self.eval_step(data, img_height=img_height, img_width=img_width, gen_cano_mesh=gen_cano_mesh)
            print ('Elapsed time for inference: {}'.format(time.time() - t))

            if 'posed_mesh' in eval_step_dict.keys() and mesh_dir is not None:
                eval_step_dict['posed_mesh'].export(os.path.join(mesh_dir, '{:06d}_implicit.ply'.format(data['inputs.frame_idx'].item())))

            if 'refined_smpl' in eval_step_dict.keys() and mesh_dir is not None:
                eval_step_dict['refined_smpl'].export(os.path.join(mesh_dir, '{:06d}_smpl.ply'.format(data['inputs.frame_idx'].item())))

            for k, v in eval_step_dict.items():
                if k not in ['output_image', 'gt_image', 'output_normal', 'output_smpl', 'output_normal_cano_front', 'output_normal_cano_back', 'posed_mesh', 'refined_smpl']:
                    eval_list[k].append(v)

            if tf_logger is not None and 'output_image' in  eval_step_dict.keys():
                tf_logger.add_image('output_image', eval_step_dict['output_image'], it + d_idx, dataformats='HWC')
                # Also save png images to image_dir
                if image_dir is not None:
                    png_file = os.path.join(image_dir, '{:06d}_view{:04d}.png'.format( data['inputs.frame_idx'].item(), int(val_dataset.cam_names[data['inputs.cam_idx'].item()]) ))
                    cv2.imwrite(png_file, cv2.cvtColor(eval_step_dict['output_image'].detach().cpu().numpy(), cv2.COLOR_RGB2BGR))

            if tf_logger is not None and 'output_normal' in  eval_step_dict.keys():
                tf_logger.add_image('output_normal', eval_step_dict['output_normal'], it + d_idx, dataformats='HWC')
                # Also save normal images to image_dir
                if image_dir is not None:
                    png_file = os.path.join(image_dir, 'normal_{:06d}_view{:04d}.png'.format( data['inputs.frame_idx'].item(), int(val_dataset.cam_names[data['inputs.cam_idx'].item()]) ))
                    cv2.imwrite(png_file, cv2.cvtColor((eval_step_dict['output_normal'].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

            if tf_logger is not None and 'output_normal_cano_front' in  eval_step_dict.keys():
                tf_logger.add_image('output_normal_cano_front', eval_step_dict['output_normal_cano_front'], it + d_idx, dataformats='HWC')

            if tf_logger is not None and 'output_normal_cano_back' in  eval_step_dict.keys():
                tf_logger.add_image('output_normal_cano_back', eval_step_dict['output_normal_cano_back'], it + d_idx, dataformats='HWC')

            if tf_logger is not None and 'gt_image' in  eval_step_dict.keys():
                tf_logger.add_image('gt_image', eval_step_dict['gt_image'], it + d_idx, dataformats='HWC')

            if tf_logger is not None and 'output_smpl' in  eval_step_dict.keys():
                tf_logger.add_image('output_smpl', eval_step_dict['output_smpl'], it + d_idx, dataformats='HWC')

            d_idx += 1

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict

    def evaluate_baseline(self, val_loader, it, val_dataset=None, tf_logger=None, mask_dir=None):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader for validation dataloader
            it (int): current training step
            val_dataset (torch.utils.data.Dataset): pytorch dataset for validation
            tf_logger (tensorboardX.SummaryWriter): TensorBoard writer
            mask_dir (str): directory to save projected 2D masks of 3D SMPL bounding boxes
        '''
        eval_list = defaultdict(list)

        if hasattr(val_dataset, 'img_size'):
            img_height, img_width = val_dataset.img_size
        else:
            img_height = img_width = None

        d_idx = 0
        for data in tqdm(val_loader):
            eval_step_dict = self.eval_step_baseline(data, img_height=img_height, img_width=img_width)

            for k, v in eval_step_dict.items():
                if k not in ['output_image', 'gt_image', 'output_normal', 'output_smpl', 'output_normal_cano_front', 'output_normal_cano_back', 'posed_mesh', 'refined_smpl']:
                    eval_list[k].append(v)

            if tf_logger is not None and 'output_image' in  eval_step_dict.keys():
                tf_logger.add_image('output_image', eval_step_dict['output_image'], it + d_idx, dataformats='HWC')

            if tf_logger is not None and 'output_normal' in  eval_step_dict.keys():
                tf_logger.add_image('output_normal', eval_step_dict['output_normal'], it + d_idx, dataformats='HWC')

            if tf_logger is not None and 'output_normal_cano_front' in  eval_step_dict.keys():
                tf_logger.add_image('output_normal_cano_front', eval_step_dict['output_normal_cano_front'], it + d_idx, dataformats='HWC')

            if tf_logger is not None and 'output_normal_cano_back' in  eval_step_dict.keys():
                tf_logger.add_image('output_normal_cano_back', eval_step_dict['output_normal_cano_back'], it + d_idx, dataformats='HWC')

            if tf_logger is not None and 'gt_image' in  eval_step_dict.keys():
                tf_logger.add_image('gt_image', eval_step_dict['gt_image'], it + d_idx, dataformats='HWC')

            if tf_logger is not None and 'output_smpl' in  eval_step_dict.keys():
                tf_logger.add_image('output_smpl', eval_step_dict['output_smpl'], it + d_idx, dataformats='HWC')

            if 'bbox_mask' in eval_step_dict.keys():
                cam_name = int(val_dataset.cam_names[data['inputs.cam_idx'].item()])
                if len(val_dataset.cam_names) == 17:
                    cam_name = cam_name - 1 if cam_name <= 19 else cam_name - 3
                else:
                    cam_name = cam_name - 1

                png_file = os.path.join(mask_dir, 'mask_{:06d}_view{:04d}.png'.format( data['inputs.frame_idx'].item(), cam_name ))
                cv2.imwrite(png_file, eval_step_dict['bbox_mask'] * 255)

            d_idx += 1

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict

    def train_step(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        ''' Performs an evaluation step.
        '''
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        ''' Performs  visualization.
        '''
        raise NotImplementedError
