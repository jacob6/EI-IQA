# EI-IQA
An Explicit-Implicit Dual Stream Network for Image Quality Assessment

## Abstract

> We propose a better deep learning model for image quality assessment, which based on 
explicit-implicit dual stream network. We use frequency domain features of kurtosis 
based on wavelet-transform to represent explicit features, and use spatial features 
extracted by convolutional neural network to represent implicit features. So we constructed
an explicit implicit parallel deep learning model, namely EI-IQA model. The EI-IQA model 
is based on the convolutional neural network VGGNet extracts the spatial domain features. 
And on this basis, by increasing the parallel wavelet kurtosis value frequency domain features,
the number of network layers of VGGNet is reduced, the training parameters are greatly reduced, 
and the sample requirements are reduced. In this paper, by cross-validation of different 
databases, we verified that the wavelet kurtosis feature fusion method based on deep learning
has a more complete feature extraction effect and a better generalization ability, which can
better simulate the human visual perception system and make subjective feelings closer to the human eye.

## Authors

Guangyi Yang, Xingyu Ding, Tian huang, Chen kun

## Requirement

Python 3.7
Pytorch 1.4.0
Tensorboard
Tensorflow

## Pretrained model
