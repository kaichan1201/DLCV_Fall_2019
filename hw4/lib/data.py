import os
import cv2
import torch
import numpy as np
import pandas as pd

import torchvision.transforms as transforms
from torch.utils.data import Dataset

from .reader import getVideoList, readShortVideo

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class TrimmedVideoDataset(Dataset):
    def __init__(self, args, mode='train', model_type='RNN', frame_num=0):
        self.mode = mode
        self.model_type = model_type

        self.downsample = args.downsample
        self.rescale = args.rescale
        self.data_dir = args.data_dir
        self.infer_data_dir = args.infer_data_dir
        if frame_num > 0:
            self.frame_num = frame_num
        else:
            self.frame_num = args.frame_num

        if self.mode == 'test':
            assert len(args.csv_path)

        if len(args.csv_path):
            self.csv_path = args.csv_path
        else:
            self.csv_path = os.path.join(self.data_dir, 'label', 'gt_{}.csv'.format(self.mode))
        self.video_dict = getVideoList(self.csv_path)

        ''' set up image transform '''
        self.transform = transforms.Compose([
                           transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                           transforms.Normalize(MEAN, STD)
                           ])

    def __len__(self):
        return len(self.video_dict['Video_index'])

    def __getitem__(self, idx):
        if len(self.infer_data_dir):
            video_path = self.infer_data_dir
        else:
            video_path = os.path.join(self.data_dir, 'video', self.mode)
        frames = readShortVideo(video_path=video_path,
                                video_category=self.video_dict['Video_category'][idx],
                                video_name=self.video_dict['Video_name'][idx],
                                downsample_factor=self.downsample,
                                rescale_factor=self.rescale)

        if self.model_type == 'CNN':
            sample_idx = np.random.randint(frames.shape[0], size=self.frame_num)
            frames = frames[sample_idx]
        elif self.model_type == 'RNN':
            if frames.shape[0] > self.frame_num:
                idx_list = np.rint(np.arange(self.frame_num) / self.frame_num * frames.shape[0]).astype(int)
                frames = frames[idx_list]

        frames_t = []
        for i in range(frames.shape[0]):
            frames_t.append(self.transform(frames[i]))

        if self.mode == 'test':
            return torch.stack(frames_t, dim=0)
        else:
            return torch.stack(frames_t, dim=0), torch.tensor(int(self.video_dict['Action_labels'][idx])).long()


class LongVideoDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode

        self.rescale = args.rescale
        self.data_dir = args.data_dir
        self.frame_num = args.frame_num

        self.txt_list_dir = os.path.join(self.data_dir, 'labels', self.mode)
        self.txt_list = os.listdir(self.txt_list_dir)
        self.video_list_dir = os.path.join(self.data_dir, 'videos', self.mode)
        self.video_list = os.listdir(self.video_list_dir)

        video_dir = os.path.join(self.video_list_dir, self.txt_list[0][:-4])
        f = cv2.imread(os.path.join(video_dir, os.listdir(video_dir)[0]))
        h, w, _ = f.shape

        ''' set up image transform '''
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((round(h*self.rescale), round(w*self.rescale))),
            transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.txt_list)

    def __getitem__(self, idx):
        txt_file = self.txt_list[idx]
        with open(os.path.join(self.txt_list_dir, txt_file)) as f:
            txt = f.readlines()

        video_dir = os.path.join(self.video_list_dir, txt_file[:-4])
        frame_id_list = sorted(os.listdir(video_dir))
        if len(frame_id_list) > self.frame_num:
            idx_list = np.rint(np.arange(self.frame_num) / self.frame_num * len(frame_id_list)).astype(int)
        else:
            idx_list = np.arange(len(frame_id_list)).astype(int)

        frames = []
        lbls = []
        for i in idx_list:
            f = cv2.imread(os.path.join(video_dir, frame_id_list[i]))
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(f))
            lbls.append([int(txt[i])])

        return torch.stack(frames, dim=0), torch.tensor(lbls).long()


class LongVideoDataset_Infer(Dataset):
    def __init__(self, args, mode='valid'):
        self.mode = mode

        self.rescale = args.rescale
        self.frame_num = args.frame_num

        self.video_list_dir = args.infer_data_dir
        self.video_list = os.listdir(self.video_list_dir)

        video_dir = os.path.join(self.video_list_dir, self.video_list[0])
        f = cv2.imread(os.path.join(video_dir, os.listdir(video_dir)[0]))
        h, w, _ = f.shape

        ''' set up image transform '''
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((round(h*self.rescale), round(w*self.rescale))),
            transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_dir = os.path.join(self.video_list_dir, self.video_list[idx])
        frame_id_list = sorted(os.listdir(video_dir))
        # if len(frame_id_list) > self.frame_num:
        #     idx_list = np.rint(np.arange(self.frame_num) / self.frame_num * len(frame_id_list)).astype(int)
        # else:
        #     idx_list = np.arange(len(frame_id_list)).astype(int)
        idx_list = np.arange(len(frame_id_list)).astype(int)

        frames = []
        for i in idx_list:
            f = cv2.imread(os.path.join(video_dir, frame_id_list[i]))
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(f))

        return torch.stack(frames, dim=0), self.video_list[idx]


if __name__ == "__main__":
    file = 'train.csv'
    df = pd.read_csv(file)
    print(df.shape)
    print(df.iloc[:, 0])
