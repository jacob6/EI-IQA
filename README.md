# EI-IQA

![framework](https://github.com/jacob6/EI-IQA/tree/master/pics/Framework.jpg)


Cite this article
Yang, G., Ding, X., Huang, T. et al. Explicit-implicit dual stream network for image quality assessment. J Image Video Proc. 2020, 48 (2020). https://doi.org/10.1186/s13640-020-00538-y

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

Guangyi Yang, Xingyu Ding, Tian huang, Chen kun, Weizheng Jin

## Training

`CUDA_VISIBLE_DEVICES=0 python VGG13plus_IQAmain.py --exp_id=0 --database=LIVE --model=VGG13plus_IQA`. 

Before training , the `im_dir` and `datainfo` in `config.yaml` must be specified

## Visualization

```
tensorboard --logdir=tensorboard_logs --port=6006
```
tensorboard_logs is your tensorboard_logs file's address

## Requirement

* Python 3.7. 
* Pytorch 1.3.0. 
* Tensorboard 1.13.1. 
* Tensorflow 1.13.1. 
* pytorch/ignite 0.2.0. 

__Note__:you need to install right CUDA version

## Pretrained model

We have trained a model on the whole LIVE dataset and you can get it in https://pan.baidu.com/s/1NaR5KvYgJ5tpotpzjRZcVw (password:nfw2). 
you can load it by `model.load_state_dict(torch.load(args.model_file))` in code

## Experiments

This table shows the SROCC values of EI-IQA and several classical IQA methods on the LIVE dataset

Algorithm | JP2K | JPEG | Noise | Blur | FF | ALL
---- | ---- | ---- | ---- | ---- | ---- | ---- |
PSNR | 0.868 | 0.885 | 0.943 | 0.761 | 0.891 | 0.866 |
SSIM | 0.938 | 0.947 | 0.964 | 0.907 | 0.956 | 0.913 |
VIF | 0.952 | 0.910 | 0.984 | 0.972 | 0.963 | 0.952 |
BIQI | 0.802 | 0.874 | 0.958 | 0.821 | 0.730 | 0.824 |
DIIVINE | 0.913 | 0.910 | 0.984 | 0.921 | 0.863 | 0.916 |
BLINDS-II | 0.951 | 0.941 | 0.978 | 0.944 | 0.927 | 0.920 |
BRISQUE | 0.947 | 0.925 | 0.989 | 0.951 | 0.903 | 0.947 |
SSEQ | 0.946 | 0.951 | 0.978 | 0.948 | 0.904 | 0.935 |
EI-IQA | 0.974 | 0.986 | 0.992 | 0.985 | 0.874 | 0.958 |

You can get more esperiments result and more details of experiments in the paper https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-020-00538-y
