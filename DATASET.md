# Dataset Preparation
## Prerequisite
For dataset preparation and preprocessing, we assume you already installed the conda environment and finished installation of SMPL model. For data preprocessing, you also need to download [this .npy file](https://github.com/zju3dv/EasyMocap/blob/98a229f2ab7647f14ac9693eab00639337274b49/data/smplx/J_regressor_body25.npy) from the [EasyMocap Repository](https://github.com/zju3dv/EasyMocap), and put it under `body_models/misc/`

## ZJU-MoCap
To download the ZJU-MoCap dataset please get access [here](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset).  After you get the dataset, extract the dataset to an arbitrary directory, denoted as ${ZJU_ROOT}. It should have the following structure: 
```
${ZJU_ROOT}
 ├-- CoreView_313
 ├-- CoreView_315
 |   ...
 └-- CoreView_394
```

To preprocess one sequence (e.g. CoreView_313), run the following
```
export PYTHONPATH=${PWD}    # only need to run this line once
python preprocess_datasets/preprocess_ZJU-MoCap.py --data-dir ${ZJU_ROOT} --out-dir ${OUTPUT_DIR} --seqname CoreView_313
```
where ${OUTPUT_DIR} is the directory where you want to save the preprocessed data. After this, create a symbolic link under `./data` directory by:
```
ln -s ${OUTPUT_DIR} data/zju_mocap
```

## H36M
Our H36M models are trained with the preprocessed version of H36M data from [Animatable NeRF](https://github.com/zju3dv/animatable_nerf). Please first email the authors of Animatable NeRF to get a copy of their preprocessed version of H36M data.

After you get the data, extract the dataset to an arbitrary directory, denoted as ${H36M_ROOT}. It should have the following structure: 
```
${H36M_ROOT}
 ├-- S1
 ├-- S2
 |   ...
 └-- S11
```

To preprocess one sequence (e.g. S9), run the following
```
export PYTHONPATH=${PWD}    # only need to run this line once
python preprocess_datasets/preprocess_H36M.py --data-dir ${H36M_ROOT} --out-dir ${OUTPUT_DIR} --seqname S9
```
where ${OUTPUT_DIR} is the directory where you want to save the preprocessed data. After this, create a symbolic link under `./data` directory by:
```
ln -s ${OUTPUT_DIR} data/h36m
```

## AIST++
Note that for preprocessing AIST++, we assume you already setup ZJU-MoCap dataset as instructed, and a symbolic link to the dataset exists as `./data/zju_mocap`. You only need motion data from AIST++, please first get the motion data from the [official website](https://google.github.io/aistplusplus_dataset/download.html).

After you get the data, extract the dataset to an arbitrary directory, denoted as ${AIST_ROOT}. It should have the following structure: 
```
${AIST_ROOT}
 └-- motions 
```

To transfer poses of sequence `gBR_sBM_cAll_d04_mBR1_ch06` of AIST++ to subject `CoreView_377` of ZJU-MoCap, run the following
```
export PYTHONPATH=${PWD}    # only need to run this line once
python preprocess_datasets/preprocess_aist.py --data-dir ${AIST_ROOT}/motions --seqname gBR_sBM_cAll_d04_mBR1_ch06 --subject CoreView_377
```
by default the data will be saved to respective directories under `./data/odp`.

## People Snapshot & AMASS-MPI-Limits
Coming soon!
