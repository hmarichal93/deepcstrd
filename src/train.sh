#!/bin/bash

### Pinus taeda v1
# No augmentation
python src/train.py --train_dataset_dir /data/maestria/resultados/deep_cstrd/pinus_v1/train \
        --val_dataset_dir /data/maestria/resultados/deep_cstrd/pinus_v1/val --logs_dir src/runs/pinus_v1_40_train_12_val


# 20 % augmented data with elastic deformation only.
 python src/train.py --dataset_dir /data/maestria/resultados/deep_cstrd/pinus_v1/  --logs_dir src/runs/pinus_v1_40_train_12_val_augmented --augmentation 1

###
 python src/train.py --dataset_dir ~/datasets/deep_cstrd/pinus_v2/  --logs_dir src/runs/pinus_v2
  python src/train.py --dataset_dir ~/datasets/deep_cstrd/pinus_v2/  --logs_dir src/runs/pinus_v2_debug


### Gleditsia triacanthos v1
python src/train.py --dataset_dir /data/maestria/resultados/deep_cstrd/gleditsia/  --logs_dir src/runs/gleditsia_v1_7_train_1_val


### Salix glauca
#### 1. Prepare dataset. Remove background and generate pith csv file
python src/preparing_dataset/salix_glauca.py

#### 2. Train model
python src/train.py --dataset_dir /data/maestria/resultados/deep_cstrd/salix_glauca/  --logs_dir src/runs/salix_glauca


