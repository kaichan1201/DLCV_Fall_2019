import os
import cv2
import torch
import numpy as np
import pandas as pd

import torchvision.transforms as transforms
from torch.utils.data import Dataset

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


class FaceDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.data_dir = args.data_dir
        self.df = pd.read_csv(os.path.join(self.data_dir, '{}.csv'.format(self.mode)))
        self.img_list = self.df['image_name']
        self.lbl_list = self.df['Smiling']

        ''' set up image transform '''
        self.transform = transforms.Compose([
                           transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                           transforms.Normalize(MEAN, STD)
                           ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.mode, self.img_list[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), torch.tensor(self.lbl_list[idx]).long()


class DigitDataset(Dataset):
    def __init__(self, args, name, mode='train'):
        self.name = name  # either 'mnistm' or 'svhn'
        self.mode = mode

        if args.infer_data_dir:
            dirs = args.infer_data_dir.split('/')
            self.data_dir = dirs[0]
            for i in range(1, len(dirs)-3):
                self.data_dir = os.path.join(self.data_dir, dirs[i])
        else:
            self.data_dir = args.data_dir
        self.df = pd.read_csv(os.path.join(self.data_dir, self.name, '{}.csv'.format(self.mode)))
        self.img_list = self.df['image_name']
        self.lbl_list = self.df['label']

        ''' set up image transform '''
        self.transform = transforms.Compose([
                           transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                           transforms.Normalize(MEAN, STD)
                           ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.name, self.mode, self.img_list[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.mode == 'test':
            return self.transform(img), torch.tensor(self.lbl_list[idx]).long(), self.img_list[idx]
        return self.transform(img), torch.tensor(self.lbl_list[idx]).long()


class DigitDataset_INFER(Dataset):
    def __init__(self, args):
        # self.name = name  # either 'mnistm' or 'svhn'
        self.data_dir = args.infer_data_dir
        self.img_list = os.listdir(self.data_dir)

        ''' set up image transform '''
        self.transform = transforms.Compose([
                           transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                           transforms.Normalize(MEAN, STD)
                           ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.img_list[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), self.img_list[idx]


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


if __name__ == "__main__":
    file = 'train.csv'
    df = pd.read_csv(file)
    print(df.shape)
    print(df.iloc[:, 0])
