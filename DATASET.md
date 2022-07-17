# Dataset Preparation
## Prerequisite
For dataset preparation and preprocessing, we assume you already installed the conda environment and finished installation of SMPL model. For data preprocessing, you also need to download [this .npy file](https://github.com/zju3dv/EasyMocap/blob/98a229f2ab7647f14ac9693eab00639337274b49/data/smplx/J_regressor_body25.npy) from the [EasyMocap Repository](https://github.com/zju3dv/EasyMocap), and put it under `body_models/misc/`

## ZJU-MoCap
To download the ZJU-MoCap dataset please get access [here](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset).

After you get the dataset, extract the dataset to an arbitrary directory, denoted as ${ZJU_ROOT}. It should have the following structure: 
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
Coming soon!

## AIST++
Coming soon!

## AMASS-MPI-Limits
Coming soon!
