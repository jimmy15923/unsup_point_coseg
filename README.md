# Unsupervised Point Cloud Object Co-segmentation by Co-Contrastive learning and Mutual Attention Sampling

This repository is the implementation of ICCV 2021 paper (Oral) **Unsupervised Point Cloud Object Co-segmentation by using Co-Contrastive learning and Mutual Attention Sampling**.
![teaser](figure/teaser.pdf)

## Requirements

We strongly recommand using the [Docker image](https://github.com/itailang/SampleNet/tree/master/registration#installation) provided by SampleNet [Lang, et al. CVPR'2020](https://arxiv.org/abs/1912.03663).

To install requirements:

```setup
pip install -r requirements.txt
```

## Data Preparation
Download the ScanObjectNN [here](https://github.com/hkust-vgd/scanobjectnn) and S3DIS [here](https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip).

And run the pre-process data to generate the S3DIS object dataset.

```python
python data_preprocess/parse_data.py
```


## Training

To train the model in the paper, run these commands, `obj` from 1 to 14 stands for each object category in ScanObjectNN:

```bash
python train.py --config=configs/scanobj.yaml --obj=1
```

## Test
Run the trained model for inference

```bash
python test.py --config=work_dirs/raw/scanobj/chair.yaml 
```

## Visualization
To genereate the GIF file in the README.md, run the command. Please note that only open3D with local monitor is supported.

```bash
python visualize.py
```
