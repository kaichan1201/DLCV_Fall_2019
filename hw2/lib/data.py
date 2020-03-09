import os
import cv2
import numpy as np
import scipy.misc

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset

# TODO: calculate mean and std of the dataset
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


class DATA(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.data_dir = args.data_dir
        self.img_dir = os.path.join(self.data_dir, self.mode, 'img')
        self.seg_dir = os.path.join(self.data_dir, self.mode, 'seg')

        self.data = list(os.listdir(self.img_dir))

        ''' set up image transform '''
        if self.mode == 'train':
            self.transform = transforms.Compose([
                               transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])

        elif self.mode == 'val' or self.mode == 'test':
            self.transform = transforms.Compose([
                               transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data[idx])
        seg_path = os.path.join(self.seg_dir, self.data[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg = cv2.imread(seg_path, 0)

        if self.mode == 'train':
            # random horizontal flip
            if np.random.random_sample() > 0.5:
                img = cv2.flip(img, 1)
                seg = cv2.flip(seg, 1)
            # random resize
            r = np.random.uniform(0.5, 1.5)
            img = resize(img, ratio=r, interpolation=cv2.INTER_LINEAR)
            seg = resize(seg, ratio=r, interpolation=cv2.INTER_NEAREST)

        return self.transform(img), torch.Tensor(seg).long()


class INFER_DATA(Dataset):
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.data = list(os.listdir(self.data_dir))
        ''' set up image transform '''
        self.transform = transforms.Compose([
                           transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                           transforms.Normalize(MEAN, STD)
                           ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.data[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), self.data[idx]


def resize(img, ratio, interpolation=cv2.INTER_LINEAR):
    """re-scale the img while maintaining the same image size"""

    new_img = cv2.resize(img, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=interpolation)

    h, w = img.shape[:2]
    new_h, new_w = new_img.shape[:2]

    if ratio > 1:
        out_img = new_img[(new_h//2 - h//2):(new_h//2 + (h-h//2)), (new_w//2 - w//2):(new_w//2 + (w-w//2))]
    else:
        out_img = np.zeros_like(img)
        out_img[(h//2 - new_h//2):(h//2 + new_h-new_h//2), (w//2 - new_w//2):(w//2 + new_w-new_w//2)] = new_img

    assert out_img.shape == img.shape
    return out_img


if __name__ == "__main__":
    test_img = cv2.imread('hw2_data/train/img/0000.png')
    big_img = resize(test_img, 1.5)
    small_img = resize(test_img, 0.7)
    cv2.imshow('img', test_img)
    cv2.imshow('big_img', big_img)
    cv2.imshow('small_img', small_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
