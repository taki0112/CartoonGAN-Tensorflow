# CartoonGAN-Tensorflow
Simple Tensorflow implementation of [CartoonGAN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf) (CVPR 2018)

## Pytorch version
* [CartoonGAN-Pytorch](https://github.com/znxlwm/pytorch-CartoonGAN)

## Requirements
* Tensorflow 1.8
* Python 3.6

## Usage
### 1. Download vgg19
* [vgg19.npy](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs)

### 2. Do edge_smooth
```
> python edge_smooth.py --dataset face2anime --img_size 256
```

```
├── dataset
   └── YOUR_DATASET_NAME
       ├── trainA
           ├── xxx.jpg (name, format doesn't matter)
           ├── yyy.png
           └── ...
       ├── trainB
           ├── zzz.jpg
           ├── www.png
           └── ...
       ├── trainB_smooth (After you run the above code, it will be created automatically)
           ├── zzz.jpg 
           ├── www.png
           └── ...
       └── testA
           ├── aaa.jpg 
           ├── bbb.png
           └── ...
```

### 3. Train
* python main.py --phase train --dataset face2anime --epoch 100 --init_epoch 1

### 4. Test
* python main.py --phase test --dataset face2anime

## Author
Junho Kim
