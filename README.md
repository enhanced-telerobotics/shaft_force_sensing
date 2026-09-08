# Shaft-integrated Force Sensing with Transformer-based Dynamics Compensation for Telesurgery

[Project Page](https://enhanced-telerobotics.github.io/shaft_force_sensing/) | 
[Paper](https://ieeexplore.ieee.org/document/11547236) | 
[Video](https://youtu.be/pcKunl9hvOI?si=YFEZaZ0H5BoHgDw0) | 
[Dataset](https://zenodo.org/records/21036880)


## Overview

This repository implements models to estimate Cartesian end-effector forces for cable-driven surgical tools using dVRK joint states and HEX10 force/torque readings. Primary contributions:

- A Transformer-based model for Cartesian force prediction (`shaft_force_sensing/models/transformer.py`).
- A learning-based dynamics compensation baseline using an LSTM that predicts internal joint torques and converts compensated torques to Cartesian forces (`shaft_force_sensing/models/lstm.py`).
- Additional backbones: Liquid Time-Constant networks (`shaft_force_sensing/models/ltc.py`) and fully connected networks (`shaft_force_sensing/models/fcn.py`).
- Utilities for data loading (`shaft_force_sensing/data/dataset.py`), training (`shaft_force_sensing/training/trainer.py`), and evaluation (`shaft_force_sensing/training/predictor.py`).

Training and experiment logs are saved to `logs/` and can be visualized with TensorBoard.

## Quick Start

1. Clone the repo and set up the environment (Python 3.10+ recommended):

```bash
git clone https://github.com/enhanced-telerobotics/shaft_force_sensing.git
cd shaft_force_sensing
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

2. Prepare the dataset (see Dataset section).
3. Run training, for example the Transformer baseline (see Training section for full commands).

## Dataset

Two options to obtain the dataset:

### Option 1 — Extract from raw bag files
1. Download the raw ROS bag files from the [dataset](https://zenodo.org/records/21036880).
2. Use the extraction tools (https://github.com/enhanced-telerobotics/dvrk_bag_extraction/tree/force_data) to convert bags into the repo format. Example:

```bash
python extract.py --bag 'Free/*' --out ./data/Automated
python extract.py --bag 'Palpation/*' --out ./data/Automated
python extract.py --bag 'Traction/*' --out ./data/Automated
python extract.py --bag 'Teleop_compressed/*' --out ./data/Teleop
```

Note: large raw image messages are excluded; compressed reference videos are provided instead. Contact the authors for access to original raw images if needed.

### Option 2 — Download extracted dataset
Download the pre-extracted [dataset](https://zenodo.org/records/21036880/files/data.tar.gz) and unpack to reproduce the following structure:

```
data/
├── Automated/
│   ├── Free_xxxx/
│   │   ├── *.csv
│   │   └── *.npz
│   ├── Palpation_xxxx/
│   └── Traction_xxxx/
└── Teleop/
    ├── C1_F_xxxx/
    │   ├── *.csv
    │   ├── *.npz
    │   ├── image_l_30hz.mp4
    │   └── image_r_30hz.mp4
    └── ...
```

Notes:

1. Automated: three load conditions (Free, Palpation, Traction). See the paper for details.
2. Teleop: three robot configurations (C1, C2, C3) and three contact conditions (Free, Rigid, Soft).
3. `.csv` files contain synchronized, preprocessed signals at 100 Hz suitable for training and evaluation. `.npz` files contain derived quantities (e.g., spatial Jacobian, rotation matrices) used by the LSTM method.

## Training

Example commands are provided in the `training/` folder. Representative commands:

### Transformer model
```bash
shaft_force_sensing/training/train_predict.sh \
  --save_dir logs/transformer/base \
  --lr 1e-3 \
  --batch_size 256 \
  --hidden_size 64 \
  --model_type transformer \
  --weight_decay 1e-4 \
  --max_epochs 100
```

### Finetune example (Transformer)
Fine-tune a pretrained Transformer on the Teleop dataset (example for model index 0):
```bash
shaft_force_sensing/training/train_predict.sh \
  --save_dir logs/transformer/ft_m0 \
  --max_epochs 100 \
  --model_idx 0 \
  --model_type transformer \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --stride 5 \
  --teleop \
  --finetune \
  --model_dir logs/transformer/base
```

### LSTM model
```bash
shaft_force_sensing/training/train_predict.sh \
  --save_dir logs/lstm/base \
  --lr 5e-3 \
  --batch_size 256 \
  --hidden_size 128 \
  --model_type lstm \
  --weight_decay 1e-6 \
  --stride 5 \
  --sequence_length 1000 \
  --max_epochs 200
```

### FCN model
Train the fully-connected baseline on the Automated dataset:
```bash
shaft_force_sensing/training/train_predict.sh \
  --save_dir logs/fcn/base \
  --lr 1e-3 \
  --batch_size 256 \
  --model_type fcn \
  --weight_decay 1e-4 \
  --sequence_length 1 \
  --max_epochs 100
```

### Evaluation
```bash
# Iteratively evaluate saved transformer models
python -m shaft_force_sensing.evaluation.batch_eval logs/transformer
```

## Citation
If you use this code for research, please cite:

```
@ARTICLE{yang2026shaft,
  author={Yang, Shuyuan and Boone, Grant and Markert, Timo and Matich, Sebastian and Theissler, Andreas and Atzmueller, Martin and Chua, Zonghe},
  journal={IEEE Transactions on Medical Robotics and Bionics}, 
  title={Shaft-Integrated Force Sensing with Transformer-Based Dynamics Compensation for Telesurgery},
  year={2026},
  volume={8},
  number={3},
  pages={970-981},
  keywords={End effectors;Image sensors;Estimation;Mechanical cables;Medical robotics;Telerobotics;Deep learning;Force sensing;telerobotics;telesurgery;deep learning},
  doi={10.1109/TMRB.2026.3699214}
}
```

## Acknowledgements
The [HEX10](https://www.resense.io/fileadmin/06-Downloads/Resense/Flyer/Resense-Flyer-F_T-Sensor-hex10.pdf) force/torque sensor was provided by [Resense GmbH, Germany](https://www.resense.io/en-en/).

The LSTM torque compensation baseline was adopted from [An Effectiveness Study Across Baseline and Learning-based Force Estimation Methods on the da Vinci Research Kit Si System](https://github.com/vu-maple-lab/dvrk_force_estimation).

## Contact
For dataset access, questions, or collaboration please open an issue.
