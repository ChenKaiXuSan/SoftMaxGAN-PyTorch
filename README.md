# SoftMaxGAN-PyTorch
PyTorch implements a SoftMax GAN neural network structure
## Overview
This repository contains an Pytorch implementation of SoftMax GAN.
With full coments and my code style.

## About SoftMax GAN
If you're new to SoftMax GAN, here's an abstract straight from the paper[1]:

Softmax GAN is a novel variant of Generative Adversarial Network (GAN). The key idea of Softmax GAN is to replace the classification loss in the original GAN with a softmax cross-entropy loss in the sample space of one single batch.

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
For the 10k epochs training on the CIFAR10 dataset, compare with about 10k samples, I get the FID: 
>   123.1316002259374

> :warning: I dont konw if the FID is right or not, because I cant get the lowwer score like the paper or the other people get it. 
## Network structure
``` python

```
## Result
- MNIST  
<!-- ![9900_MNSIT](img/9900_MNIST.png) -->
- CIFAR10  
<!-- ![9900_cifar10](img/9900_cifar10.png) -->
- Fashion-MNIST
<!-- ![9900_fashion](img/9900_fashion.png) -->
## Reference
<!-- 1. [SAGAN](http://arxiv.org/abs/1805.08318) -->