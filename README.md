## Weakly Supervised Contrastive Learning (ICCV2021)


This repository contains PyTorch evaluation code, training code and pretrained models for WCL.

For details see [Weakly Supervised Contrastive Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Zheng_Weakly_Supervised_Contrastive_Learning_ICCV_2021_paper.pdf) by Mingkai Zheng, Fei Wang, Shan You, Chen Qian, Changshui Zhang, Xiaogang Wang and Chang Xu

![WCL](img/framework.png)

## Train
```
python train.py --gpu_ids 1
```
## Reproducing
To run the code, you probably need to change the Dataset setting (dataset/cifar.py), and Pytorch DDP setting (util/dist_init.py) for your own server enviroments.

In this code, we adopt a hidden dimension of 4096 and output dimension 256 for the projection head (we use 2048 and 128 in our paper) since we found the performance can be further improved a little bit.

|          |Arch | BatchSize | Epochs | Linear Eval | Linear Eval (Paper) | Download  |
|----------|:----:|:---:|:---:|:---:|:---:|:---:|
|  WCL | ResNet50 | 256 | 100  | - % | - % | [wcl-100.pth](https://drive.google.com/file/d/1WaLP5oAE9-_zre21wdV-C8ox0_YVVi8p/view?usp=share_link) |
|  WCL | ResNet50 | 256 | 200  | -% | - % | [wcl-300.pth](https://drive.google.com/file/d/1VDG2EWtK4Wm-dTdclVfx5X6AWg3aI2Yn/view?usp=share_link) |
|  WCL | ResNet50 | 256 | 200  | -% | - % | [wcl-400.pth](https://drive.google.com/file/d/1vSlT0_EoJqfg66euyvJHah6Y0sq5bHwh/view?usp=share_link) |

If you want to test the pretained model, please download the weights from the link above, and move it to the checkpoints folder (create one if you don't have .checkpoints/ directory). The evaluation scripts also has been provided in script/train.sh


## Citation
If you find that wcl interesting and help your research, please consider citing it:
```
@InProceedings{Zheng_2021_ICCV,
    author    = {Zheng, Mingkai and Wang, Fei and You, Shan and Qian, Chen and Zhang, Changshui and Wang, Xiaogang and Xu, Chang},
    title     = {Weakly Supervised Contrastive Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {10042-10051}
}
```

