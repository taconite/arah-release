import numpy as np
import pickle as pkl

import torch
import torch.nn as nn

from human_body_prior.body_model.lbs import lbs

class BodyModel(nn.Module):

    def __init__(self,
                 bm_path,
                 params=None,
                 num_betas=10,
                 batch_size=1, v_template=None,
                 use_posedirs=True,
                 dtype=torch.float32):

        super(BodyModel, self).__init__()

        '''
        :param bm_path: path to a SMPL model as pkl file
        :param num_betas: number of shape parameters to include.
                if betas are provided in params, num_betas would be overloaded with number of thoes betas
        :param batch_size: number of smpl vertices to get
        :param device: default on gpu
        :param dtype: float precision of the compuations
        :return: verts, trans, pose, betas
        '''
        # Todo: if params the batchsize should be read from one of the params

        self.dtype = dtype

        if params is None: params = {}

        # -- Load SMPL params --
        if '.npz' in bm_path:
            smpl_dict = np.load(bm_path, encoding='latin1')
        elif '.pkl' in bm_path:
            with open(bm_path, 'rb') as f:
                smpl_dict = pkl.load(f, encoding='latin1')
        else:
            raise ValueError('bm_path should be either a .pkl nor .npz file')

        njoints = smpl_dict['posedirs'].shape[2] // 3
        self.model_type = {69: 'smpl'}[njoints]

        assert self.model_type in ['smpl'], ValueError(
            'model_type should be in smpl.')

        # Mean template vertices
        if v_template is None:
            v_template = np.repeat(smpl_dict['v_template'][np.newaxis], batch_size, axis=0)
        else:
            v_template = np.repeat(v_template[np.newaxis], batch_size, axis=0)

        self.register_buffer('v_template', torch.tensor(v_template, dtype=dtype))

        self.register_buffer('f', torch.tensor(smpl_dict['f'].astype(np.int32), dtype=torch.int32))

        if len(params):
            if 'betas' in params.keys():
                num_betas = params['betas'].shape[1]

        num_total_betas = smpl_dict['shapedirs'].shape[-1]
        if num_betas < 1:
            num_betas = num_total_betas

        print (smpl_dict['shapedirs'].shape)
        shapedirs = smpl_dict['shapedirs'][:, :, :num_betas]
        self.register_buffer('shapedirs', torch.tensor(shapedirs, dtype=dtype))

        # Regressor for joint locations given shape - 6890 x 24
        self.register_buffer('J_regressor', torch.tensor(smpl_dict['J_regressor'].toarray(), dtype=dtype))

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
        if use_posedirs:
            posedirs = smpl_dict['posedirs']
            posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
            self.register_buffer('posedirs', torch.tensor(posedirs, dtype=dtype))
        else:
            self.posedirs = None

        # indices of parents for each joints
        kintree_table = smpl_dict['kintree_table'].astype(np.int32)
        self.register_buffer('kintree_table', torch.tensor(kintree_table, dtype=torch.int32))

        # LBS weights
        # weights = np.repeat(smpl_dict['weights'][np.newaxis], batch_size, axis=0)
        weights = smpl_dict['weights']
        self.register_buffer('weights', torch.tensor(weights, dtype=dtype))

        if 'trans' in params.keys():
            trans = params['trans']
        else:
            trans = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('trans', nn.Parameter(trans, requires_grad=True))

        # root_orient
        root_orient = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('root_orient', nn.Parameter(root_orient, requires_grad=True))

        # pose_body
        if 'pose_body' in params.keys():
            pose_body = params['pose_body']
        else:
            pose_body = torch.tensor(np.zeros((batch_size, 63)), dtype=dtype, requires_grad=True)
        self.register_parameter('pose_body', nn.Parameter(pose_body, requires_grad=True))

        # pose_hand
        if 'pose_hand' in params.keys():
            pose_hand = params['pose_hand']
        else:
            pose_hand = torch.tensor(np.zeros((batch_size, 1 * 3 * 2)), dtype=dtype, requires_grad=True)
        self.register_parameter('pose_hand', nn.Parameter(pose_hand, requires_grad=True))

        if 'betas' in params.keys():
            betas = params['betas']
        else:
            betas = torch.tensor(np.zeros((batch_size, num_betas)), dtype=dtype, requires_grad=True)
        self.register_parameter('betas', nn.Parameter(betas, requires_grad=True))

        self.batch_size = batch_size

    def forward(self, root_orient=None, pose_body=None, pose_hand=None, betas=None,
                trans=None, return_dict=False, v_template=None, clothed_v_template=None, **kwargs):
        ''' Forward function

        Args:
            root_orient (Tensor): root orientation/global rotation in axis-angle form
            pose_body (Tensor): rotations of body joints in axis-angle form
            pose_hand (Tensor): rotations of hands in axis-angle form
            betas (Tensor): body shape parameters
            trans (Tensor): global translation
            return_dict (bool): return a dictionary instad of object
            v_template (Tensor): mean-shape template, if betas is None
            clothed_v_template (Tensor): clothed shape in A-pose, with pose-dependent cloth-deformations
        '''
        assert not (v_template  is not None and betas  is not None), ValueError('vtemplate and betas could not be used jointly.')
        assert self.model_type in ['smpl'], ValueError(
            'model_type should be in smpl')

        if root_orient is None:  root_orient = self.root_orient
        if pose_body is None:  pose_body = self.pose_body
        if pose_hand is None:  pose_hand = self.pose_hand

        if pose_hand is None:  pose_hand = self.pose_hand

        if trans is None: trans = self.trans
        if v_template is None: v_template = self.v_template
        if betas is None: betas = self.betas

        if v_template.size(0) != pose_body.size(0):
            v_template = v_template[:pose_body.size(0)] # this is fine since actual batch size will
                                                        # only be equal to or less than specified batch
                                                        # size

        full_pose = torch.cat([root_orient, pose_body, pose_hand], dim=1)

        shape_components = betas
        shapedirs = self.shapedirs

        verts, joints, joints_a_pose, bone_transforms, abs_bone_transforms, v_posed = lbs(betas=shape_components, pose=full_pose, v_template=v_template,
                                                                                          clothed_v_template=clothed_v_template,
                                                                                          shapedirs=shapedirs, posedirs=self.posedirs,
                                                                                          J_regressor=self.J_regressor, parents=self.kintree_table[0].long(),
                                                                                          lbs_weights=self.weights,
                                                                                          dtype=self.dtype)

        Jtr = joints + trans.unsqueeze(dim=1)
        verts = verts + trans.unsqueeze(dim=1)
        v_posed = v_posed + trans.unsqueeze(dim=1)

        res = {}
        res['v'] = verts
        res['v_a_pose'] = v_posed
        res['f'] = self.f
        res['abs_bone_transforms'] = abs_bone_transforms
        res['bone_transforms'] = bone_transforms
        res['betas'] = betas
        res['Jtr'] = Jtr  # Todo: ik can be made with vposer
        res['Jtr_a_pose'] = joints_a_pose

        res['pose_body'] = pose_body
        res['full_pose'] = full_pose

        if not return_dict:
            class result_meta(object):
                pass

            res_class = result_meta()
            for k, v in res.items():
                res_class.__setattr__(k, v)
            res = res_class

        return res


