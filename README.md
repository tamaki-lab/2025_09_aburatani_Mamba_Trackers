# Mamba (Selective State Space Model) for Multi-object tracking 

<div align="center">

**Language**: English | [简体中文](README_CN.md)

</div>

## ✅Introduction

This repo is the ***unofficial*** implementation of the following Mamba-based Multi-object trackers: (This two papers have not released the official code yet)

1. MambaTrack: A Simple Baseline for Multiple Object Tracking with State Space Model [ACM MM'25](http://arxiv.org/abs/2408.09178)

2. TrackSSM: A General Motion Predictor by State-Space Model [arXiv:2409.00487](http://arxiv.org/abs/2409.00487)

## 🗺️RoadMap

- [] Add model of paper: Exploring Learning-based Motion Models inMulti-Object Tracking [arXiv:2403.10826](http://arxiv.org/abs/2403.10826)

## 🏃Results & Checkpoints

- **Supporting dataset: DanceTrack, MOT17 and VisDrone2019**

Demo result of DanceTrack-val:

***MambaTrack:***

![alt text](assets/mambatrack_demo.gif "title")

***TrackSSM:***

![alt text](assets/trackssm_demo.gif "title")

- **Results of DacneTrack-val and checkpoints:**

| Models | HOTA | MOTA | IDF1| checkpoint |
| ------ | ------ | ------ | ------ | ------ |
| MambaTrack | 32.672 | 78.392 | 26.419 | [Baidu Disk](https://pan.baidu.com/s/1QuMQV3iubDIkDUMExDcSOQ), code: e0mv | 
| TrackSSM | 27.536| 72.366 | 20.756 | [Baidu Disk](https://pan.baidu.com/s/1hHOkvhmICRC2zB0A-SJyDQ), code: 2797| 

> I'm trying to achieve better results

## 📑Dataset Preparation

### Training

For training the mamba-based models, all data is converted to the trajectory format by `tools/gen_traj_data.py`. 

MOT17 dataset, run:

```bash
python tools/gen_traj_data.py --mot17 --save_name mot17
```

DanceTrack dataset, run:

```bash
python tools/gen_traj_data.py --dancetrack --save_name dancetrack
```

VisDrone dataset, run:

```bash
python tools/gen_traj_data.py --visdrone --save_name visdrone
```

**Remember to modify the `DATA_ROOT` according to your autual path**

After running the code, a json file will be generated in `ssm_tracker/traj_anno_data`.

### Inference

For inference, we **preprocess** the detection result first.

First things first, organize all the video frames in the subfoler `images`, and then `test` (or `val`), following the format: (same as YOLO format)

```
DanceTrack
    |
    |____images
            |_____test
                    |_____dancetrack0001
                                |______xxxx.jpg
```

Then run the yolox detector:

```bash
python tools/gen_det_results.py --dataset_name dancetrack --data_root /data/datasets/DanceTrack/images/ --split val --exp_file yolox_exps/custom/yolox_x.py --model_path weights/yolox_dancetrack.pth.tar --generate_meta_data
```

> you can refer to `experiments/gen_yolox_det_results.sh`

> The pretrained YOLOX-X model of DanceTrack can be downloaded from their huggingface repo. The detection results are provided in `./det_results`


## 🔍Models & Guidelines

All mamba-like models are under `./ssm_tracker`, and all kalman-like models are under `./kalman_tracker`.

### 1. MambaTrack:

The architecture of MambaTrack is as follows:

![alt text](assets/mambatrack.png "title")

The corresponding config file is `ssm_tracker/cfgs/MambaTrack.yaml`

### 2. TrackSSM:

The architecture of TrackSSM is as follows:

![alt text](assets/trackssm.png "title")

The corresponding config file is `ssm_tracker/cfgs/TrackSSM.yaml`

### 3. Training

For training, please first modify the corresponding 'true' and 'false' in the bash file `experiments/train_ssm_tracker.sh`, and run:

```bash
sh experiments/train_ssm_tracker.sh
```

### 4. Inference

For testing, please first modify the corresponding 'true' and 'false' in the bash file `experiments/inference_ssm_tracker.sh`, and modify the following arguments:

1. `--det_path`: the detection result files path, contains `{seq_name}.txt`
2. `--motion_model_path`: the trained checkpoint path
3. `--config_file `: same as train
4. `--data_root`: your data root of the dataset, following the yolo format

Then run

```bash
sh experiments/train_ssm_tracker.sh
```