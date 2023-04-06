# Neural Correspondence Field for Object Pose Estimation

This repository provides the source code and trained models of the 6D object pose estimation method presented in:

**Neural Correspondence Field for Object Pose Estimation**<br>
European Conference on Computer Vision (ECCV) 2022<br>
[PDF](https://arxiv.org/pdf/2208.00113.pdf) | [BIB](https://linhuang17.github.io/NCF/resources/huang2022ncf.txt) | [Project website](https://linhuang17.github.io/NCF/)

Contents: [Setup](#setup) | [Usage](#usage) | [Pre-trained models](#pre-trained-models)


## <a name="setup"></a>1. Setup

### 1.1 Cloning the repository

Download the code:
```
git clone https://github.com/LinHuang17/NCF-code.git
cd NCF-code
```

### 1.2 Python environment and dependencies

Create and activate conda environment with dependencies:
```
conda env create -f environment.yaml
conda activate ncf
```

### 1.3 BOP datasets

For experiments on existing [BOP datasets](https://bop.felk.cvut.cz/datasets/), please follow the instructions on the [website](https://bop.felk.cvut.cz/datasets/) to download the base archives, 3D object models, the training images, and the test images.

For YCB-V, you are expected to have files: `ycbv_base.zip`, `ycbv_models.zip`, `ycbv_train_pbr.zip`, `ycbv_train_real.zip` (used for training models with real images), and `ycbv_test_bop19.zip`. Then, unpack them into folder `<path/to/ycbv>`.

## <a name="usage"></a>2. Usage

### 2.1 Inference with a pre-trained model

To evaluate on an object (e.g., cracker box) from YCB-V:

First, download and unpack the [pre-trained models](#pre-trained-models) into folder `<path/to/ncf_ycbv>`.

Then, run the following command with the cracker box's pre-trained model:
```
export CUDA_VISIBLE_DEVICES=0 
python -m apps.eval --exp_id ncf_ycbv_run2_eval --work_base_path <path/to/ncf_results> --dataset ycbv --eval_data ycbv_bop_cha --model_dir <path/to/ycbv/models> --ds_ycbv_dir <path/to/ycbv> --obj_id 2 --bbx_size 380 --eval_perf True --load_netG_checkpoint_path <path/to/ncf_ycbv/ncf_ycbv_obj2> --num_in_batch 10000
```

where `work_base_path` is the path to the results (e.g., the estimated pose csv file as `ncf-obj2_ycbv-Rt-time.csv`), `model_dir` is the path to the YCB-V 3D object models, `ds_ycbv_dir` is the path to the YCB-V dataset, and `load_netG_checkpoint_path` is the path to the cracker box's pre-trained model.


### 2.2 Training your own model


## <a name="pre-trained-models"></a>3. Pre-trained models

- [YCB-V](https://drive.google.com/file/d/19rcvuIC7Ilu0MHPgLxmbxeUkOgBHR2be/view?usp=sharing)