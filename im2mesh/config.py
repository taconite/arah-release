import yaml
from yaml import Loader

from im2mesh import data
from im2mesh import metaavatar_render

method_dict = {
    'metaavatar_render': metaavatar_render,
}

# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model(cfg, dataset=None, val_size=None, mode='train'):
    ''' Returns the pytorch_lightning.LightningModule instance.

    Args:
        cfg (dict): config dictionary
        dataset (dataset): dataset object
    '''
    method = cfg['method']
    model = method_dict[method].config.get_model(cfg, dataset=dataset, mode=mode)

    lightning_model = method_dict[method].lightning_model.LightningModel(
        model=model,
        cfg=cfg,
        val_size=val_size,
    )
    return lightning_model


# Datasets
def get_dataset(mode, cfg, view_split=None, subsampling_rate=None, start_frame=None, end_frame=None):
    ''' Returns the dataset.

    Args:
        mode (str): which mode the dataset is. Can be either train, val or test
        cfg (dict): config dictionary
        view_split (list of str): which view(s) to use. If None, will load all views
        subsampling_rate (int): frame subsampling rate for the dataset
        start_frame (int): starting frame
        end_frame (int): ending frame
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    use_aug = cfg['data']['use_aug']
    normalized_scale = cfg['data']['normalized_scale']

    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }

    split = splits[mode]

    if view_split is None:
        view_splits = {
            'train': cfg['data']['train_views'],
            'val': cfg['data']['val_views'],
            'test': cfg['data']['test_views'],
        }

        view_split = view_splits[mode]

    if subsampling_rate is None:
        subsampling_rates = {
            'train': cfg['data']['train_subsampling_rate'],
            'val': cfg['data']['val_subsampling_rate'],
            'test': cfg['data']['test_subsampling_rate'],
        }

        subsampling_rate = subsampling_rates[mode]

    if start_frame is None:
        start_frames = {
            'train': cfg['data']['train_start_frame'],
            'val': cfg['data']['val_start_frame'],
            'test': cfg['data']['test_start_frame'],
        }

        start_frame = start_frames[mode]

    if end_frame is None:
        end_frames = {
            'train': cfg['data']['train_end_frame'],
            'val': cfg['data']['val_end_frame'],
            'test': cfg['data']['test_end_frame'],
        }

        end_frame = end_frames[mode]

    # Create dataset
    if dataset_type == 'people_snapshot':
        num_fg_samples = cfg['data']['num_fg_samples']
        num_bg_samples = cfg['data']['num_bg_samples']

        off_surface_thr = cfg['data']['off_surface_thr']
        inside_thr = cfg['data']['inside_thr']
        box_margin = cfg['data']['box_margin']
        sampling = cfg['data']['sampling']
        erode_mask = cfg['data']['erode_mask']
        sample_reg_surface = cfg['data']['sample_reg_surface']

        inside_weight = cfg['training']['inside_weight']

        high_res = cfg['data']['high_res']

        dataset = data.PeopleSnapshotDataset(
            dataset_folder=dataset_folder,
            subjects=split,
            mode=mode,
            img_size=(540, 540) if not high_res or mode in ['val', 'test'] else (1080, 1080),
            num_fg_samples=num_fg_samples,
            num_bg_samples=num_bg_samples,
            sampling_rate=subsampling_rate,
            start_frame=start_frame,
            end_frame=end_frame,
            off_surface_thr=off_surface_thr,
            inside_thr=inside_thr,
            box_margin=box_margin,
            sampling=sampling,
            sample_reg_surface=sample_reg_surface,
            sample_inside=inside_weight > 0,
            erode_mask=erode_mask,
        )
    elif dataset_type == 'zju_mocap':
        num_fg_samples = cfg['data']['num_fg_samples']
        num_bg_samples = cfg['data']['num_bg_samples']

        off_surface_thr = cfg['data']['off_surface_thr']
        inside_thr = cfg['data']['inside_thr']
        box_margin = cfg['data']['box_margin']
        sampling = cfg['data']['sampling']
        erode_mask = cfg['data']['erode_mask']
        sample_reg_surface = cfg['data']['sample_reg_surface']

        inside_weight = cfg['training']['inside_weight']

        high_res = cfg['data']['high_res']

        dataset = data.ZJUMOCAPDataset(
            dataset_folder=dataset_folder,
            subjects=split,
            mode=mode,
            img_size=(512, 512) if not high_res or mode in ['val', 'test'] else (1024, 1024),
            num_fg_samples=num_fg_samples,
            num_bg_samples=num_bg_samples,
            sampling_rate=subsampling_rate,
            start_frame=start_frame,
            end_frame=end_frame,
            views=view_split,
            off_surface_thr=off_surface_thr,
            inside_thr=inside_thr,
            box_margin=box_margin,
            sampling=sampling,
            sample_reg_surface=sample_reg_surface,
            sample_inside=inside_weight > 0,
            erode_mask=erode_mask,
        )
    elif dataset_type == 'h36m':
        num_fg_samples = cfg['data']['num_fg_samples']
        num_bg_samples = cfg['data']['num_bg_samples']

        off_surface_thr = cfg['data']['off_surface_thr']
        inside_thr = cfg['data']['inside_thr']
        box_margin = cfg['data']['box_margin']
        sampling = cfg['data']['sampling']
        erode_mask = cfg['data']['erode_mask']
        sample_reg_surface = cfg['data']['sample_reg_surface']

        inside_weight = cfg['training']['inside_weight']

        dataset = data.H36MDataset(
            dataset_folder=dataset_folder,
            subjects=split,
            mode=mode,
            img_size=(1002, 1000),
            num_fg_samples=num_fg_samples,
            num_bg_samples=num_bg_samples,
            sampling_rate=subsampling_rate,
            start_frame=start_frame,
            end_frame=end_frame,
            views=view_split,
            off_surface_thr=off_surface_thr,
            inside_thr=inside_thr,
            box_margin=box_margin,
            sampling=sampling,
            sample_reg_surface=sample_reg_surface,
            sample_inside=inside_weight > 0,
            erode_mask=erode_mask,
        )
    elif dataset_type == 'zju_mocap_odp':
        num_fg_samples = cfg['data']['num_fg_samples']
        num_bg_samples = cfg['data']['num_bg_samples']

        box_margin = cfg['data']['box_margin']
        pose_dir = cfg['data']['pose_dir']

        dataset = data.ZJUMOCAPODPDataset(
            dataset_folder=dataset_folder,
            subjects=split,
            pose_dir=pose_dir,
            mode=mode,
            img_size=(512, 512),
            num_fg_samples=num_fg_samples,
            num_bg_samples=num_bg_samples,
            sampling_rate=subsampling_rate,
            start_frame=start_frame,
            end_frame=end_frame,
            views=view_split,
            box_margin=box_margin
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset
