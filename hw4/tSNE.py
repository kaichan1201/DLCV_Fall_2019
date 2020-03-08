import os
import torch
import numpy as np
from sklearn.manifold import TSNE

import lib.model as model
import lib.parser as parser
import lib.data as data

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

th = 1000

CNN_frame_num = 2
RNN_frame_num = 15


if __name__ == '__main__':

    args = parser.arg_parse()

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    if not os.path.exists('vis'):
        os.makedirs('vis')

    # prepare data
    print("Preparing data...")
    CNN_val_loader = torch.utils.data.DataLoader(data.TrimmedVideoDataset(args, mode='valid', model_type='CNN',
                                                                          frame_num=CNN_frame_num),
                                                 batch_size=1,
                                                 num_workers=args.workers,
                                                 shuffle=False)
    RNN_val_loader = torch.utils.data.DataLoader(data.TrimmedVideoDataset(args, mode='valid', model_type='RNN',
                                                                          frame_num=RNN_frame_num),
                                                 batch_size=1,
                                                 num_workers=args.workers,
                                                 shuffle=False)

    # prepare model
    print("Preparing model...")
    extractor = model.E_Resnet50(pretrained=True)
    RNN_model = model.RNN()
    checkpoint = torch.load(args.pretrained)
    RNN_model.load_state_dict(checkpoint['RNN'])
    gru = RNN_model.gru

    if use_cuda:
        extractor.cuda()
        gru.cuda()

    extractor.eval()
    gru.eval()

    print("Extracting CNN features...")
    CNN_feat_list = []
    gt_list = []
    for idx, (videos, gt) in enumerate(CNN_val_loader):
        if idx >= th:
            break
        with torch.no_grad():
            if use_cuda:
                videos, gt = videos.cuda(), gt.cuda()
            for i in range(CNN_frame_num):
                feat = extractor(videos[:, i, :, :, :])
            feat = feat.view(-1)
            CNN_feat_list.append(feat.cpu().numpy())
            gt_list.append(gt.item())

    CNN_feat_list = np.array(CNN_feat_list)
    gt_list = np.array(gt_list)

    print("Applying tSNE...")
    CNN_feat_tsne = TSNE(n_components=2).fit_transform(CNN_feat_list)

    print("Plotting...")
    plt.figure()
    colors = cm.rainbow(np.linspace(0, 1, 11))
    for i, c in enumerate(colors):
        points = CNN_feat_tsne[gt_list == i]
        plt.scatter(points[:, 0], points[:, 1], c=c, s=5)
    plt.title('tSNE_CNN')
    plt.savefig(os.path.join('vis', 'tSNE_CNN.jpg'))
    plt.close()

    print("Extracting RNN features...")
    RNN_feat_list = []
    gt_list = []
    for idx, (video, gt) in enumerate(RNN_val_loader):
        if idx >= th:
            break
        with torch.no_grad():
            if use_cuda:
                video, gt = video.cuda(), gt.cuda()  # video: (1, T, C, H, W)
            video = video.squeeze(0)  # video: (T, C, H, W)

            '''extracting features'''
            feat = extractor(video)  # (T, feat_size)
            r_out, _ = gru(feat.unsqueeze(0))
            out_feat = r_out[:, -1, :].view(-1)
            RNN_feat_list.append(out_feat.cpu().numpy())
            gt_list.append(gt.item())

    RNN_feat_list = np.array(RNN_feat_list)
    gt_list = np.array(gt_list)

    print("Applying tSNE...")
    RNN_feat_tsne = TSNE(n_components=2).fit_transform(RNN_feat_list)

    print("Plotting...")
    plt.figure()
    colors = cm.rainbow(np.linspace(0, 1, 11))
    for i, c in enumerate(colors):
        points = RNN_feat_tsne[gt_list == i]
        plt.scatter(points[:, 0], points[:, 1], c=c, s=5)
    plt.title('tSNE_RNN')
    plt.savefig(os.path.join('vis', 'tSNE_RNN.jpg'))
    plt.close()


