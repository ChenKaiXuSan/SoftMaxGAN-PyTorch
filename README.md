# SoftMaxGAN-PyTorch
PyTorch implements a SoftMax GAN neural network structure
## Overview
This repository contains an Pytorch implementation of SoftMax GAN.
With full coments and my code style.

## About SAGAN
If you're new to SNGAN, here's an abstract straight from the paper[1]:

In this paper, we propose the Self-Attention Gen- erative Adversarial Network (SAGAN) which allows attention-driven, long-range dependency modeling for image generation tasks. Traditional convolutional GANs generate high-resolution de- tails as a function of only spatially local points in lower-resolution feature maps. In SAGAN, de- tails can be generated using cues from all feature locations. Moreover, the discriminator can check that highly detailed features in distant portions of the image are consistent with each other. Fur- thermore, recent work has shown that generator conditioning affects GAN performance. Leverag- ing this insight, we apply spectral normalization to the GAN generator and find that this improves training dynamics. The proposed SAGAN per- forms better than prior work1, boosting the best published Inception score from 36.8 to 52.52 and reducing Fr´echet Inception distance from 27.62 to 18.65 on the challenging ImageNet dataset. Visu- alization of the attention layers shows that the generator leverages neighborhoods that correspond to object shapes rather than local regions of fixed shape.

## Dataset 
- MNIST
`python3 main.py --dataset mnist --channels 1`
- FashionMNIST
`python3 main.py --dataset fashion --channels 1`
- Cifar10
`python3 main.py --dataset cifar10 --channels 3`

## Implement
``` python

```

## Usage
- MNSIT
`python3 main.py --dataset mnist --channels 1 --version [version] --batch_size [] >logs/[log_path]`
- FashionMNIST
`python3 main.py --dataset fashion --channels 1 --version [version] --batch_size [] >logs/[log_path]`
- Cifar10
`python3 main.py --dataset cifar10 --channels 3 -version [version] --batch_size [] >logs/[log_path]`

## FID
FID is a measure of similarity between two datasets of images. It was shown to correlate well with human judgement of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks. FID is calculated by computing the Fréchet distance between two Gaussians fitted to feature representations of the Inception network.

For the FID, I use the pytorch implement of this repository. [FID score for PyTorch](https://github.com/mseitzer/pytorch-fid)

- MNIST
For the 10k epochs training on MNIST dataset, compare with about 10k samples, I get the FID: 
> 85.71858261109799
- CIFAR10
<!-- For the 10k epochs training on the CIFAR10 dataset, compare with about 10k samples, I get the FID: 
> 108.10053254296571 :warning: I think this test is failing, the reason dont konw why. -->
- FASHION-MNIST
<!-- For the 10k epochs training on the CIFAR10 dataset, compare with about 10k samples, I get the FID: 
>  46.96466240507351 -->

> :warning: I dont konw if the FID is right or not, because I cant get the lowwer score like the paper or the other people get it. 
## Network structure
``` python
Generator(
  (l1): Sequential(
    (0): ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (l2): Sequential(
    (0): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (l3): Sequential(
    (0): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (l4): Sequential(
    (0): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (last): Sequential(
    (0): ConvTranspose2d(64, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): Tanh()
  )
  (attn2): Attention(
    (theta): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (phi): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (g): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (o): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)
Discriminator(
  (l1): Sequential(
    (0): Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
  )
  (l2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
  )
  (l3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
  )
  (l4): Sequential(
    (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
  )
  (attn2): Attention(
    (theta): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (phi): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (g): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (o): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (last_adv): Sequential(
    (0): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
  )
)
```
## Result
- MNIST  
<!-- ![9900_MNSIT](img/9900_MNIST.png) -->
- CIFAR10  
<!-- ![9900_cifar10](img/9900_cifar10.png) -->
- Fashion-MNIST
<!-- ![9900_fashion](img/9900_fashion.png) -->
## Reference
1. [SAGAN](http://arxiv.org/abs/1805.08318)