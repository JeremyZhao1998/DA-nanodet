# DA-nanodet: Domain Ddaptive nanodet
Domain adaptive object detection based on lightweight [nanodet](https://github.com/RangiLyu/nanodet) detector.

Code developed based on nanodet. Project page: https://github.com/RangiLyu/nanodet

We perform unsupervised domain adaptation, i.e. trained on labeled source domain and ublabeled target domain, and test on target domain.



## Install

### Requirements

- Linux or MacOS
- CUDA >= 10.2
- Python >= 3.7
- Pytorch >= 1.10.0, <2.0.0

### Step

1. Create a conda virtual environment and then activate it.

```bash
 conda create -n nanodet python=3.8 -y
 conda activate nanodet
```

1. Install pytorch

```bash
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge
```

1. Clone this repository

```bash
git clone https://github.com/JeremyZhao1998/DA-nanodet.git
cd DA-nanodet
```

1. Install requirements

```bash
pip install -r requirements.txt
```

1. Setup NanoDet

```bash
python setup.py develop
```



## Model

We test our method on lightweight setting of NanoDet-m-0.5x ShuffleNetV2 0.5x model, the original setting and nanodet pretrained weights are listed:

| Model          | Backbone          | Resolution | FLOPS | Params | Pre-trained weight                                           |
| -------------- | ----------------- | ---------- | ----- | ------ | ------------------------------------------------------------ |
| NanoDet-m-0.5x | ShuffleNetV2 0.5x | 448*256    | 0.3G  | 0.28M  | [download](https://drive.google.com/file/d/1rMHkD30jacjRpslmQja5jls86xd0YssR/view?usp=sharing) |

Our domain adaptation method invloves additional components only during training. Test time parameter size and inference speed are kept the same.



## Dataset

We use the autonomous driving images captured in sunny daytime as source dataset, and use the overcast daytime, light rain daytime, heavy rain daytime and clear night as target dataset.



## Training

1. **Start training**

   NanoDet is now using [pytorch lightning](https://github.com/PyTorchLightning/pytorch-lightning) for training.

   For both single-GPU or multiple-GPUs, run:

   ```
   python tools/train.py CONFIG_FILE_PATH
   ```

2. **Visualize Logs**

   TensorBoard logs are saved in `save_dir` which you set in config file.

   To visualize tensorboard logs, run:

   ```
   cd <YOUR_SAVE_DIR>
   tensorboard --logdir ./
   ```