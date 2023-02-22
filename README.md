# ARAH: Animatable Volume Rendering of Articulated Human SDFs
## [Paper](https://drive.google.com/file/d/10yCrdOadwKNiDQBni23_W03ZwVafkfCJ/view?usp=sharing) | [Project Page](https://neuralbodies.github.io/arah/)

<img src="assets/mono.gif" width="800"/> 

This repository contains the implementation of our paper
[ARAH: Animatable Volume Rendering of Articulated Human SDFs]().

You can find detailed usage instructions for using pretrained models and training your own models below.

If you find our code useful, please cite:

```bibtex
@inproceedings{ARAH:2022:ECCV,
  title = {ARAH: Animatable Volume Rendering of Articulated Human SDFs},
  author = {Shaofei Wang and Katja Schwarz and Andreas Geiger and Siyu Tang},
  booktitle = {European Conference on Computer Vision},
  year = {2022}
}
```

## Installation
### Environment Setup
This repository has been tested on the following platform:
1) Python 3.9.7, PyTorch 1.10 with CUDA 11.3 and cuDNN 8.2.0, Ubuntu 20.04/CentOS 7.9.2009

To clone the repo, run either:
```
git clone --recursive https://github.com/taconite/arah-release.git
```
or
```
git clone https://github.com/taconite/arah-release.git
git submodule update --init --recursive
```

Next, you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `arah` using
```
conda env create -f environment.yml
conda activate arah
```

Lastly, compile the extension modules. You can do this via
```
python setup.py build_ext --inplace
```

### SMPL Setup
Download `SMPL v1.0 for Python 2.7` from [SMPL website](https://smpl.is.tue.mpg.de/) (for male and female models), and `SMPLIFY_CODE_V2.ZIP` from [SMPLify website](https://smplify.is.tue.mpg.de/) (for the neutral model). After downloading, inside `SMPL_python_v.1.0.0.zip`, male and female models are `smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl` and `smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl`, respectively. Inside `mpips_smplify_public_v2.zip`, the neutral model is `smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`. Remove the chumpy objects in these .pkl models using [this code](https://github.com/vchoutas/smplx/tree/master/tools) under a Python 2 environment (you can create such an environment with conda). Finally, rename the newly generated .pkl files and copy them to subdirectories under `./body_models/smpl/`. Eventually, the `./body_models` folder should have the following structure:
```
body_models
 └-- smpl
    ├-- male
    |   └-- model.pkl
    ├-- female
    |   └-- model.pkl
    └-- neutral
        └-- model.pkl

```

Then, run the following script to extract necessary SMPL parameters used in our code:
```
python extract_smpl_parameters.py
```
The extracted SMPL parameters will be saved into `./body_models/misc/`.

## Quick Demo on the [AIST++](https://google.github.io/aistplusplus_dataset/factsfigures.html) Dataset
1. Run `bash download_demo_data.sh` to download and extract 1) pretrained models and 2) the preprocessed AIST++ sequence.
2. Run the pre-trained model on AIST++ poses via
    ```
    python test.py --num-workers 4 configs/arah-zju/ZJUMOCAP-377-mono_4gpus.yaml
    ```
    The script will compose a result .mp4 video in `out/arah-zju/ZJUMOCAP-377-mono_4gpus/vis`. There are a total of 258 frames, so it will take some time to render all the frames. If you want to check the result quickly run:
    ```
    python test.py --num-workers 4 --end-frame 10 configs/arah-zju/ZJUMOCAP-377-mono_4gpus.yaml
    ```
    to render only the first 10 frames, or
    ```
    python test.py --num-workers 4 --subsampling-rate 25 configs/arah-zju/ZJUMOCAP-377-mono_4gpus.yaml
    ```
    to render every 25th frame. Inference requires ~20GB VRAM, if you don't have so much memory, add `--low-vram` option. This should run with ~12GB VRAM at the cost of longer inference time. 

## Results on ZJU-MoCap
For easy comparison to our approach, we also store all our rendering and geometry reconstruction results on the ZJU-MoCap dataset [here](https://drive.google.com/file/d/14Icwr85NmQozfQOdZAMX5T8QF7tb8vsc/view?usp=share_link). Train/val splits on cameras/poses follow [NeuralBody's split](https://github.com/zju3dv/neuralbody/blob/master/supplementary_material.md#training-and-test-data). Pseudo ground truths for geometry reconstruction on the ZJU-MoCap dataset are stored in [this folder](https://drive.google.com/drive/folders/1-OE3h7nxg7ixL3yh0Y7bGYKVsNWS-Zm4?usp=share_link). For evaluation script and data split of geometry reconstruction please refer to [this comment](https://github.com/taconite/arah-release/issues/9#issuecomment-1359209138).

## Dataset preparation
Due to license issues, we cannot publicly distribute our preprocessed ZJU-MoCap and H36M data. You have to get the raw data from their respective sources and use our preprocessing script to generate data that is suitable for our training/validation scripts. Please follow the steps in [DATASET.md](DATASET.md).

## Download pre-trained skinning and SDF networks
We provide pre-trained [models](https://drive.google.com/drive/folders/1nraph3_QeCeKU4reFd_OgJA696-dLjYh?usp=sharing) on the CAPE dataset as prerequisites, including 1) meta learned skinning network on the CAPE dataset, 2) MetaAvatar SDF model. After downloading them, please put them in respective folders under `./out/meta-avatar`.

## Training
To train new networks from scratch, run
```
python train.py --num-workers 4 ${path_to_config}
```
Where ${path_to_config} is the relative path to the yaml config file, e.g. config/arah-zju/ZJUMOCAP-313_4gpus.yaml

Training and validation use [wandb](https://wandb.ai/site) for logging, which is free to use but requires online register.

Note that by default, all models are trained on 4 GPUs with a total batch size of 4. You can change the value of `training.gpus` to `[0]` in the configuration file to train on a single GPU with a batch size of 1, however the model accuracy may drop and the training might become less stable.

## Pre-trained models of ARAH (Work In Progress)
We provide pre-trained [models](https://drive.google.com/drive/folders/1ZlIvwdfHdDsdGW-6YmeDF8znvbxRgiYK?usp=sharing), including multi-view and monocular models. After downloading them, please put them in respective folders under `./out/arah-zju` or `./out/arah-h36m`.

## Validate the trained model on within-distribution poses
To validate the trained model on novel views of training poses, run
```
python validate.py --novel-view --num-workers 4 ${path_to_config}
```
To validate the trained model on novel views of unseen poses, run
```
python validate.py --novel-pose --num-workers 4 ${path_to_config}
```

## Test the trained model on out-of-distribution poses
To run the trained model on preprocessed poses, run
```
python test.py --num-workers 4 --pose-dir ${pose_dir} --test-views ${view} configs/arah/${config}
```
where `${pose_dir}` denotes the directory under `data/odp/CoreView_${sequence_name}/` that contains target (out-of-distribution) poses. `${view}` indicates the testing views from which to render the model.

Currently, the code only supports animating ZJU-MoCap models for out-of-distribution poses.

## License
We employ [MIT License](LICENSE.md) for the ARAH code, which covers
```
extract_smpl_parameters.py
train.py
validate.py
test.py
setup.py
configs
im2mesh/
preprocess_datasets/preprocess_ZJU-MoCap.py
```
Our SDF network is based on [SIREN](https://github.com/vsitzmann/siren). Our mesh extraction code is borrowed from [DeepSDF](https://github.com/facebookresearch/DeepSDF). The structure of our rendering code is largely based on [IDR](https://github.com/lioryariv/idr). Our root-finding code is modified from [SNARF](https://github.com/xuchen-ethz/snarf). We thank authors of these papers for their wonderful works which inspired this paper.

Modules not covered by our license are:
1) Modified code from [EasyMocap](https://github.com/zju3dv/EasyMocap) to preprocess ZJU-MoCap/H36M datasets (`./preprocess_datasets/easymocap`);
2) Modified code from [SMPL-X](https://github.com/nghorbani/human_body_prior) (`./human_body_prior`);

for these parts, please consult their respective licenses and cite the respective papers.
