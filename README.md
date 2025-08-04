# KETI_DVS_VFI

## Getting Started
### Installation
For environment setup and installation instructions, please refer to the TimeLens-XL GitHub repository(https://github.com/OpenImagingLab/TimeLens-XL).
Our code is based on their implementation.

### Download Pretrained Model & bsergb dataset
1. Download Link : [x5 interpolation weights](https://drive.google.com/file/d/11HxbvZ2_VmQZhHHXv7shCWhM75Y8pZj2/view?usp=sharing)
```bash
$ mkdir weights
$ cd weights
```
Place downloaded weights under "./weights" folder.

2. Download Link: [Public Dataset_BSERGB](https://drive.google.com/file/d/1DPY0G1sr2TfP_Pt0rZpSgmCNIwbwtbNe/view?usp=sharing)

 ```bash
$ cd dataset
```
Place downloaded dataset under "./dataset" folder.

Modify the dataset path in params/Paths/BDERGB.py to point to your ```Local PATH```.

3. Download Link: [Dists pytorch weight](https://github.com/dingkeyan93/DISTS/blob/master/DISTS_pytorch/weights.pt)
```bash
$ cd ./losses/DISTS/DISTS_pytorch
```
Place downloaded weights.pt under "./losses/DISTS/DISTS_pytorch" folder.

Modify the dataset path in params/Paths/BDERGB.py to point to your ```Local PATH```.

## Inference model (x5 interpolation)
```bash
$ python run_network.py --param_name traintest_BSERGB_x5AdamwithLPIPS --model_name Expv8_large --model_pretrained <pretrained_model_weight path> --skip_training
```




