# CartoonGAN-Tensorflow
Simple Tensorflow implementation of [CartoonGAN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf) (CVPR 2018)

## Requirements
* Tensorflow 1.8
* Python 3.6

## Usage
### Do edge_smooth
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
       ├── trainB_blur (After you run the above code, it will be created automatically)
           ├── zzz.jpg 
           ├── www.png
           └── ...
       └── testA
           ├── aaa.jpg 
           ├── bbb.png
           └── ...
```

### Train
* python main.py --phase train --dataset face2anime --epoch 20 --init_epoch 1

### Test
* python main.py --phase test --dataset face2anime

## Author
Junho Kim
