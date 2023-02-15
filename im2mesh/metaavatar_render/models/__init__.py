import torch
import numpy as np
import torch.nn as nn
import im2mesh.utils.sdf_meshing as sdf_meshing

from im2mesh.metaavatar_render.renderer.implicit_differentiable_renderer import IDHRNetwork
from im2mesh.metaavatar_render.renderer.ray_tracing import BodyRayTracing

from im2mesh.utils.root_finding_utils import (
    normalize_canonical_points, unnormalize_canonical_points
)

from im2mesh.utils.utils import augm_rots

from human_body_prior.body_model.lbs import lbs

class MetaAvatarRender(nn.Module):
    ''' MetaAvatar model class.
    '''
    def __init__(
            self,
            sdf_decoder=None,
            skinning_model=None,
            color_decoder=None,
            deviation_decoder=None,
            train_cameras=False,
            train_smpl=False,
            train_latent_code=False,
            train_geo_latent_code=False,
            cano_view_dirs=True,
            near_surface_samples=16,
            far_surface_samples=16,
            train_skinning_net=False,
            render_last_pt=False,
            pose_input_noise=True,
            view_input_noise=True,
            nv_noise_type='rotation',
            low_vram=False,
            n_steps=64,
            **kwargs):
        ''' Initialization of the MetaAvatarRender class.

        Args:
            sdf_decoder (torch.nn.Module): SDF network
            skinning_model (torch.nn.Module): skinning network
            color_decoder (torch.nn.Module): color network
            deviation_decoder (torch.nn.Module): learnable scalar (i.e. b for volume rendering)
            train_cameras (bool): whether to optimize camera extrinsics or not
            train_smpl (bool): whether to optimize SMPL parameters or not
            train_latent_code (bool): whether to employ and optimize latent codes for the color network
            train_geo_latent_code (bool): whether to employ and optimize latent codes for the SDF network
            cano_view_dirs (bool): whether to canonicalize viewing directions or not before feeding them to the color network
            near_surface_samples (int): how many near-surface points to sample
            far_surface_samples (int): how many far-surface points to sample
            train_skinning_net (bool): whether to optimize skinning network (with implicit gradients) or not
            render_last_pt (bool): if set to True, distance of the last point on ray will be set to 1e10, forcing the volume renderer
                                   to render the last point, regardless of its density
            pose_input_noise (bool): whether to add noise to pose inputs during training
            view_input_noise (bool): whether to add noise to viewing directions during training
            nv_noise_type (str): which type of augmentation noise to use for view and normals
            low_vram (bool): if set to True, use less VRAM for inference
            n_steps (int): how many points to sample for rays that do not intersect with surface
        '''
        super().__init__()

        self.sdf_decoder = sdf_decoder
        self.skinning_model = skinning_model

        self.color_decoder = color_decoder
        self.deviation_decoder = deviation_decoder
        self.pose_input_noise = pose_input_noise
        self.view_input_noise = view_input_noise
        self.nv_noise_type = nv_noise_type

        ray_tracer = BodyRayTracing(root_finding_threshold=1e-5, n_steps=n_steps, near_surface_vol_samples=near_surface_samples, far_surface_vol_samples=far_surface_samples, sample_bg_pts=0, low_vram=low_vram)
        self.idhr_network = IDHRNetwork(deviation_decoder, color_decoder, skinning_model, ray_tracer, cano_view_dirs=cano_view_dirs, train_skinning_net=train_skinning_net, render_last_pt=render_last_pt, low_vram=low_vram)

        self.train_cameras = train_cameras
        if train_cameras:
            cam_trans = kwargs.get('cam_trans')
            cam_rots = kwargs.get('cam_rots')

            assert (cam_trans is not None and cam_rots is not None)

            self.register_parameter('cam_trans', nn.Parameter(torch.from_numpy(cam_trans)))
            self.register_parameter('cam_rots', nn.Parameter(torch.from_numpy(cam_rots)))

        self.train_smpl = train_smpl
        if train_smpl:
            # raise NotImplementedError('Training code for SMPL poses/shapes are not implemented yet!')
            # Load templates
            gender = kwargs.get('gender')
            v_template = np.load('body_models/misc/v_templates.npz')[gender]
            lbs_weights = np.load('body_models/misc/skinning_weights_all.npz')[gender]
            posedirs = np.load('body_models/misc/posedirs_all.npz')[gender]
            posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
            shapedirs = np.load('body_models/misc/shapedirs_all.npz')[gender]
            J_regressor = np.load('body_models/misc/J_regressors.npz')[gender]
            kintree_table = np.load('body_models/misc/kintree_table.npy')

            self.register_buffer('v_template', torch.tensor(v_template, dtype=torch.float32).unsqueeze(0))
            self.register_buffer('posedirs', torch.tensor(posedirs, dtype=torch.float32))
            self.register_buffer('shapedirs', torch.tensor(shapedirs, dtype=torch.float32))
            self.register_buffer('J_regressor', torch.tensor(J_regressor, dtype=torch.float32))
            self.register_buffer('lbs_weights', torch.tensor(lbs_weights, dtype=torch.float32))
            self.register_buffer('kintree_table', torch.tensor(kintree_table, dtype=torch.int32))

            root_orient = kwargs.get('root_orient')
            pose_body = kwargs.get('pose_body')
            pose_hand = kwargs.get('pose_hand')
            trans = kwargs.get('trans')
            betas = kwargs.get('betas')
            frames = kwargs.get('frames')

            self.frames = frames

            param_dict = {}
            param_dict.update({'root_orient_' + str(fr): nn.Parameter(torch.tensor(root_orient[idx], dtype=torch.float32)) for idx, fr in enumerate(frames)})
            param_dict.update({'pose_body_' + str(fr): nn.Parameter(torch.tensor(pose_body[idx], dtype=torch.float32)) for idx, fr in enumerate(frames)})
            param_dict.update({'pose_hand_' + str(fr): nn.Parameter(torch.tensor(pose_hand[idx], dtype=torch.float32)) for idx, fr in enumerate(frames)})
            param_dict.update({'trans_' + str(fr): nn.Parameter(torch.tensor(trans[idx], dtype=torch.float32)) for idx, fr in enumerate(frames)})

            self.body_poses = nn.ParameterDict(param_dict)

            self.register_parameter('betas', nn.Parameter(torch.tensor(betas, dtype=torch.float32)))

        self.train_latent_code = train_latent_code
        self.train_geo_latent_code = train_geo_latent_code

        if train_latent_code or train_geo_latent_code:
            self.latent = nn.Embedding(kwargs.get('n_data_points'), 128)
            if not train_smpl:
                frames = kwargs.get('frames')
                self.frames = frames


    def forward(self, inputs, gen_cano_mesh=False, eval=False):
        ''' Makes a forward pass through the model.

        Args:
            inputs (dict): input data
            gen_cano_mesh (bool): whether to extract canonical mesh from SDF or not (takes extra time)
            eval (bool): use evaluation mode
        '''

        rots = inputs['rots']
        Jtrs = inputs['Jtrs']

        batch_size = rots.size(0)
        device = rots.device

        decoder_input = {'coords': torch.zeros(1, 1, 3, dtype=torch.float32, device=device), 'rots': rots[0].unsqueeze(0), 'Jtrs': Jtrs[0].unsqueeze(0)}
        if 'geo_latent_code_idx' in inputs.keys():
            geo_latent_code = self.latent(inputs['geo_latent_code_idx'])
            decoder_input.update({'latent': geo_latent_code})

        # Do augmentation to input poses and views, if applicable
        with_noise = False
        if (self.pose_input_noise or self.view_input_noise) and not eval:
            if np.random.uniform() <= 0.5:
                if self.pose_input_noise:
                    decoder_input['rots_noise'] = torch.normal(mean=0, std=0.1, size=rots.shape, dtype=rots.dtype, device=device)
                    inputs['pose_cond']['rot_noise'] = torch.normal(mean=0, std=0.1, size=(batch_size,9), dtype=rots.dtype, device=device)
                    inputs['pose_cond']['trans_noise'] = torch.normal(mean=0, std=0.1, size=(batch_size,3), dtype=rots.dtype, device=device)

                if self.view_input_noise:
                    if self.nv_noise_type == 'gaussian':
                        inputs['pose_cond']['view_noise'] = torch.normal(mean=0, std=0.1, size=inputs['ray_dirs'].shape, dtype=rots.dtype, device=device)
                    elif self.nv_noise_type == 'rotation':
                        inputs['pose_cond']['view_noise'] = torch.tensor(augm_rots(45, 45, 45), dtype=torch.float32, device=device).unsqueeze(0)
                    else:
                        raise ValueError('wrong nv_noise_type, expected either gaussian or rotation, got {}'.format(self.nv_noise_type))

                with_noise = True

        # Generate SDF network from hypernetwork (MetaAvatar)
        output = self.sdf_decoder(decoder_input)
        sdf_decoder = output['decoder']
        sdf_params = output['params']

        # Prepare inputs for the skinning network
        coord_min = inputs['coord_min']
        coord_max = inputs['coord_max']
        center = inputs['center']

        vol_feat = torch.empty(batch_size, 0, device=device)
        loc = torch.zeros(batch_size, 1, 3, device=device, dtype=torch.float32)
        sc_factor = torch.ones(batch_size, 1, 1, device=device, dtype=torch.float32)

        inputs.update({'loc': loc,
                       'sc_factor': sc_factor,
                       'vol_feat': vol_feat,
                       'sdf_network': sdf_decoder,
                     })

        if 'latent_code_idx' in inputs['pose_cond'].keys():
            latent_code = self.latent(inputs['pose_cond']['latent_code_idx'])
            inputs['pose_cond']['latent_code'] = latent_code

        model_outputs = self.idhr_network(inputs)
        model_outputs.update({'sdf_params': sdf_params})

        if gen_cano_mesh:
            with torch.no_grad():
                verts, faces = sdf_meshing.create_mesh_vertices_and_faces(sdf_decoder, N=256,
                                                                          max_batch=64 ** 3)

                from im2mesh.utils.root_finding_utils import (
                    forward_skinning,
                )
                points_hat = torch.tensor(verts, dtype=torch.float32, device=device).unsqueeze(0)
                points_hat = unnormalize_canonical_points(points_hat, coord_min, coord_max, center)
                points_lbs, _ = forward_skinning(points_hat,
                                                 loc=loc,
                                                 sc_factor=sc_factor,
                                                 coord_min=coord_min,
                                                 coord_max=coord_max,
                                                 center=center,
                                                 skinning_model=self.skinning_model,
                                                 vol_feat=vol_feat,
                                                 bone_transforms=inputs['bone_transforms'],
                                                 return_w=False
                                                )


                points_bar = points_lbs + inputs['trans']

                from pytorch3d.utils import cameras_from_opencv_projection
                from pytorch3d.structures import Meshes
                from pytorch3d.renderer import (
                    look_at_view_transform,
                    PerspectiveCameras,
                    FoVPerspectiveCameras,
                    RasterizationSettings,
                    MeshRenderer,
                    MeshRasterizer,
                    TexturesVertex
                )

                faces_torch = torch.tensor(faces.copy(), dtype=torch.int64, device=device).unsqueeze(0)
                mesh_bar = Meshes(verts=points_bar, faces=faces_torch)

                # TODO: image size here should be changable
                raster_settings = RasterizationSettings(
                    image_size=(512, 512),
                )

                cam_rot = inputs['cam_rot']
                cam_trans = inputs['cam_trans']
                K = inputs['intrinsics']

                image_size = torch.tensor([[512, 512]], dtype=torch.float32, device=device)
                cameras = cameras_from_opencv_projection(cam_rot, cam_trans, K, image_size).to(device)

                rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
                rendered = rasterizer(mesh_bar)

                # Compute normal image in transformed space
                fg_mask = rendered.pix_to_face >= 0
                fg_faces = rendered.pix_to_face[fg_mask]
                faces_normals = -mesh_bar.faces_normals_packed().squeeze(0)
                normal_image = -torch.ones(1, 512, 512, 3, dtype=torch.float32, device=device)
                normal_image.masked_scatter_(fg_mask, torch.einsum('bij,pj->pi', cam_rot, faces_normals[fg_faces, :]))

                normal_image = ((normal_image + 1) / 2.0).clip(0.0, 1.0)

                model_outputs.update({'output_normal': normal_image})

                # Render normals in canonical space
                # Frontal normal
                R, T = look_at_view_transform(2.0, 0.0, 0)
                cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

                verts_torch = torch.tensor(verts.copy(), dtype=torch.float32, device=device).unsqueeze(0)
                faces_torch = torch.tensor(faces.copy(), dtype=torch.float32, device=device).unsqueeze(0)
                mesh = Meshes(verts_torch, faces_torch)

                raster_settings = RasterizationSettings(
                    image_size=512,
                )
                rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
                rendered = rasterizer(mesh)

                fg_mask = rendered.pix_to_face >= 0
                fg_faces = rendered.pix_to_face[fg_mask]
                faces_normals = mesh.faces_normals_packed().squeeze(0)
                normal_image = torch.zeros(1, 512, 512, 3, dtype=torch.float32, device=device)
                normal_image.masked_scatter_(fg_mask, faces_normals[fg_faces, :])

                normal_image = ((normal_image + 1) / 2.0).clip(0.0, 1.0)
                model_outputs.update({'normal_cano_front': normal_image})

                # Back normal
                R, T = look_at_view_transform(2.0, 0.0, 180.0)
                cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

                verts_torch = torch.tensor(verts.copy(), dtype=torch.float32, device=device).unsqueeze(0)
                faces_torch = torch.tensor(faces.copy(), dtype=torch.float32, device=device).unsqueeze(0)
                mesh = Meshes(verts_torch, faces_torch)

                rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
                rendered = rasterizer(mesh)

                fg_mask = rendered.pix_to_face >= 0
                fg_faces = rendered.pix_to_face[fg_mask]
                faces_normals = mesh.faces_normals_packed().squeeze(0)
                normal_image = torch.zeros(1, 512, 512, 3, dtype=torch.float32, device=device)
                normal_image.masked_scatter_(fg_mask, faces_normals[fg_faces, :])

                normal_image = ((normal_image + 1) / 2.0).clip(0.0, 1.0)
                model_outputs.update({'normal_cano_back': normal_image})
                # model_outputs.update({'posed_mesh': posed_mesh})
                # model_outputs.update({'refined_smpl': smpl_mesh})

        return model_outputs

    def forward_smpl(self, betas, root_orient, pose_body, pose_hand, trans):
        ''' Makes a forward pass through the SMPL LBS.

        Args:
            betas (torch.Tensor of batch_size x 10): body shape
            root_orient (torch.Tensor of batch_size x 3): global orientation
            pose_body (torch.Tensor of batch_size x 63): local body poses, excluding hands
            pose_hand (torch.Tensor of batch_size x 6): hand poses
            trans (torch.Tensor of batch_size x 3): global translation
        '''
        full_pose = torch.cat([root_orient, pose_body, pose_hand], dim=-1)
        verts_posed, Jtrs_posed, Jtrs, bone_transforms, _, minimal_shape, = lbs(betas=betas,
                                                                                pose=full_pose,
                                                                                v_template=self.v_template.clone(),
                                                                                clothed_v_template=None,
                                                                                shapedirs=self.shapedirs.clone(),
                                                                                posedirs=self.posedirs.clone(),
                                                                                J_regressor=self.J_regressor.clone(),
                                                                                parents=self.kintree_table[0].long(),
                                                                                lbs_weights=self.lbs_weights.clone(),
                                                                                dtype=torch.float32)

        return verts_posed, Jtrs, Jtrs_posed, bone_transforms, minimal_shape

    def network_parameters(self):
        for name, param in self.named_parameters():
            if name not in ['cam_rots', 'cam_trans', 'root_orient', 'pose_body', 'pose_hand', 'trans', 'betas']:
                yield param

    def camera_parameters(self):
        for name, param in self.named_parameters():
            if name in ['cam_rots', 'cam_trans']:
                yield param

    def smpl_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith( ('body_poses', 'betas') ):
                yield param
