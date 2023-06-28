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

2. Install pytorch

```bash
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge
```

3. Clone this repository

```bash
git clone https://github.com/JeremyZhao1998/DA-nanodet.git
cd DA-nanodet
```

4. Install requirements

```bash
pip install -r requirements.txt
```

5. Setup NanoDet

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

We use the autonomous driving images captured in sunny daytime as source dataset, and use the overcast daytime, light rain daytime, heavy rain daytime and clear night as target dataset. We evaluate and report the AP@50 metric.

**Sunny -> Heavy rain:**

|                       | pedestrian | vehicle | sign | arrow | mean |
| --------------------- | ---------- | ------- | ---- | ----- | ---- |
| Source only           | 16.5       | 51.5    | 30.7 | 25.8  | 31.1 |
| DA(Teacher)           | 17.7       | 62.1    | 33.1 | 30.2  | 35.8 |
| Oracle(source+target) | 42.0       | 75.2    | 42.0 | 35.5  | 48.7 |



## Training

**Start training**

NanoDet is now using [pytorch lightning](https://github.com/PyTorchLightning/pytorch-lightning) for training. 
For both single-GPU or multiple-GPUs, run:

   ```bash
   python tools/train.py CONFIG_FILE_PATH
   ```

**Visualize Logs**

   TensorBoard logs are saved in `save_dir` which you set in config file. To visualize tensorboard logs, run:

   ```bash
   cd <YOUR_SAVE_DIR>
   tensorboard --logdir ./
   ```
    
Then open `http://localhost:6006` in your browser, you can see the training curves.
