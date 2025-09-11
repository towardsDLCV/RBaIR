# Installation

### Dependencies Installation

This repository is built in PyTorch 2.4.0 and tested on Ubuntu 22.04 environment (Python3.9, CUDA12.4).
Follow these intructions

1. Clone our repository
```bash
git clone https://github.com/towardsDLCV/RBaIR.git
cd RBaIR
```

2. Create conda environment
The Conda environment used can be recreated using the env.yml file
```bash
conda create -n RBaIR python=3.9
conda activate RBaIR
pip install -r requirements.txt
```

### Dataset Download and Preperation

All the datasets for 5 tasks used in the paper can be downloaded from the following locations:

Denoising: [BSD400](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing), [WED](https://drive.google.com/file/d/1e62XGdi5c6IbvkZ70LFq0KLRhFvih7US/view?usp=sharing), [Urban100](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u), [Kodak24](https://r0k.us/graphics/kodak/), [BSD68](https://github.com/clausmichele/CBSD68-dataset/tree/master/CBSD68/original)

Deraining: [Train100L&Rain100L](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f?usp=sharing)

Dehazing: Train[ RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2), Test [SOTS-Outdoor](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)

Deblur: [GoPro](https://drive.google.com/file/d/1y_wQ5G5B65HS_mdIjxKYTcnRys_AGh5v/view?usp=sharing)

Enhance: [LOL-V1](https://daooshee.github.io/BMVC2018website/)

The training data should be placed in ``` data/Train/{task_name}``` directory where ```task_name``` can be Denoise, Derain, Dehaze, Deblur, or Enhance.

The testing data should be placed in the ```test``` directory wherein each task has a seperate directory. 

The training and test datasets are organized as:
