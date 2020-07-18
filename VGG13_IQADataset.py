import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
from scipy.signal import convolve2d
import numpy as np
import h5py
from pywt import dwt2
from scipy import stats

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'


def default_loader(path):
    return Image.open(path).convert('L')  # 读取一张图片并灰度化


def LocalNormalization(patch, P=3, Q=3, C=1):
    kernel = np.ones((P, Q)) / (P * Q)  # 滤波器k
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
    patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0)
    return patch_ln


# 图像块归一化

def wavel(img):
    k = []
    for i in range(38):
        db = 'db' + str(i + 1)
        cA, (cH, cV, cD) = dwt2(img, db)
        C = np.hstack((cA, cH, cV, cD))
        C = np.ravel(C)
        k.append([[stats.kurtosis(C)]])
    k = torch.tensor(k)
    return k


# 提取图像块的峰值特征函数

def OverlappingCropPatches(im, patch_size=32, stride=32):
    w, h = im.size
    patches = ()
    patcheswl = ()
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            patch = LocalNormalization(patch[0].numpy())
            patchwl = wavel(patch)
            patches = patches + (patch,)
            patcheswl = patcheswl + (patchwl,)
    return patches, patcheswl


# 既返回图块pathes也返回图块的小波峰值特征patheswl

class IQADataset(Dataset):
    def __init__(self, conf, EXP_ID, status='train', loader=default_loader):
        self.loader = loader
        im_dir = conf['im_dir']
        self.patch_size = conf['patch_size']  # 图块尺寸
        self.stride = conf['stride']  # 卷积步长
        datainfo = conf['datainfo']

        Info = h5py.File(datainfo, 'r')
        index = Info['index'][:, int(EXP_ID) % 1000]  #
        ref_ids = Info['ref_ids'][0, :]  # 所有参考图片序号
        test_ratio = conf['test_ratio']  # 测试集比例
        train_ratio = conf['train_ratio']  # 训练集比例
        trainindex = index[:int(train_ratio * len(index))]  # 训练部分序号
        testindex = index[int((1 - test_ratio) * len(index)):]  # 测试部分序号
        train_index, val_index, test_index = [], [], []
        for i in range(len(ref_ids)):
            train_index.append(i) if (ref_ids[i] in trainindex) else \
                test_index.append(i) if (ref_ids[i] in testindex) else \
                    val_index.append(i)
        # 将参考图像的序号加到train test val(验证集)
        if status == 'train':
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))
        if status == 'test':
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))
        if status == 'val':
            self.index = val_index
            print("# Val Images: {}".format(len(self.index)))
        # 不同状态下选择不同的数据集
        print('Index:')
        print(self.index)

        self.mos = Info['subjective_scores'][0, self.index]  # 对应数据集客观得分
        self.mos_std = Info['subjective_scoresSTD'][0, self.index]  #
        self.distortion_types = Info['distortion_types'][0, self.index]  # 对应数据集失真类型
        im_names = [Info[Info['im_names'][0, :][i]].value.tobytes() \
                        [::2].decode() for i in self.index]

        self.patches = ()
        self.patcheswl = ()
        self.label = []
        self.label_std = []
        self.label2 = []
        # 初始化各个参数label1图像分数 label1_std label2失真类型
        # 此时self.index表示所需数据集的序号
        for idx in range(len(self.index)):
            # for idx in range(10):
            # print("Preprocessing Image: {}".format(im_names[idx]))
            im = self.loader(os.path.join(im_dir, im_names[idx]))
            patches, patcheswl = OverlappingCropPatches(im, self.patch_size, self.stride)
            # x依次导入图片并j预处理
            if status == 'train':
                self.patches = self.patches + patches  # 裁剪的图块
                self.patcheswl = self.patcheswl + patcheswl  # 对应图块的小波峰值x特征
                for i in range(len(patches)):
                    self.label.append(self.mos[idx])  # 每个图块的客观分数
                    self.label_std.append(self.mos_std[idx])
                    self.label2.append(self.distortion_types[idx])  # 每个图块的失真类型
            else:
                self.patches = self.patches + (torch.stack(patches),)  #
                self.patcheswl = self.patcheswl + (torch.stack(patcheswl),)  #
                self.label.append(self.mos[idx])
                self.label_std.append(self.mos_std[idx])
                self.label2.append(self.distortion_types[idx])
                # torch.stack是指组合数据；torch.tensor指同数据类型多维矩阵

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return ((self.patches[idx], self.patcheswl[idx]), (torch.Tensor([self.label[idx], ]),
                                                           torch.Tensor([self.label_std[idx], ]),
                                                           torch.Tensor([self.label2[idx], ])))
    # 返回剪裁归一化后的图块patches；对应的小波峰值特征向量patcheswl；客观得分；；失真类型
