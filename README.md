# Towards Accurate Alignment in Real-time 3D Hand-Mesh Reconstruction



## Introduction

This is official **[PyTorch](https://pytorch.org/)** implementation of **[Towards Accurate Alignment in Real-time 3D Hand-Mesh Reconstruction (ICCV 2021)](https://wbstx.github.io/handar/)**. 

## Preparation

1. `pip install -r reqirements.txt` ⚠️ If your vispy is > 0.5.3, the code may not work.
2. Replace two files from the official vispy library with my codes in the vispy folder: vispy/io/mesh.py and vispy/io/waverfront.py. These two codes are for reading obj and mtl files.
3. Download MANO_RIGHT.pkl from [here](https://mano.is.tue.mpg.de/) and put it in common/utils/manopth/mano/models.
4. Download the [FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/) dataset and the root/bounding box prediction from [I2L-MeshNet](https://github.com/mks0601/I2L-MeshNet_RELEASE). Put them in the right palace stated by data/FreiHAND/FreiHAND.py.
5. Download the pre-trained weights from [here](https://drive.google.com/file/d/1cipSmx_iIKvdeA5yLscp1o37W40Q4ZaU/view?usp=sharing). Put it in the weights folder.

## Visualization

![visualization](https://github.com/wbstx/handAR/blob/main/figures/visualization.png?raw=true)

I implement a opencv-based visualization program to overlap the reconstructed hand mesh over the user's hand in the image space. Just simply run `python mesh_demo.py` in the test_video folder.

⚠️ This program is only tested on Windows 10. I am not sure if it works on other operating systems.

The program is easy to be modified to capture camera images.

## Dataset Testing

To test the performance on the FreiHAND dataset, run

`python -m torch.distributed.launch --nproc_per_node=1 test.py --gpu 0 --stage lixel --test_epoch 24`

And you will find the prediction result in json format in output/result.

## Network Training

To release

## Acknowledgement

The code of this work is heavily borrowed from [I2L-MeshNet](https://github.com/mks0601/I2L-MeshNet_RELEASE) and [manopth](https://github.com/hassony2/manopth). Please also refer to these amazing works.

## Reference

> ```
> @inproceedings{tang2021towards,
>   title={Towards Accurate Alignment in Real-time 3D Hand-Mesh Reconstruction},
>   author={Tang, Xiao and Wang, Tianyu and Fu, Chi-Wing},
>   booktitle={International Conference on Computer Vision (ICCV)},
>   pages={11698--11707},
>   year={2021}
> }
> ```

