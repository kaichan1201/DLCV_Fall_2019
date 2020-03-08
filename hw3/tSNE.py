import os
import torch
import numpy as np
from sklearn.manifold import TSNE

import lib.model as model
import lib.model_ADDA as model_ADDA
import lib.parser as parser
import lib.data as data

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

th = 1000


if __name__ == '__main__':

    args = parser.arg_parse()

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    if not os.path.exists('vis'):
        os.makedirs('vis')

    # prepare data
    print("Preparing data...")
    src_val_loader = torch.utils.data.DataLoader(data.DigitDataset(args, name=args.src, mode='test'),
                                                 batch_size=1,
                                                 num_workers=args.workers,
                                                 shuffle=False)
    tgt_val_loader = torch.utils.data.DataLoader(data.DigitDataset(args, name=args.tgt, mode='test'),
                                                 batch_size=1,
                                                 num_workers=args.workers,
                                                 shuffle=False)

    # prepare model
    print("Preparing model...")
    if args.tSNE_model == 'DANN':
        my_model = model.DANN()
        my_model.load_state_dict(torch.load(args.pretrained))
        extractor = my_model.extractor
    else:
        extractor = model_ADDA.E()
        checkpoint = torch.load(args.pretrained)
        extractor.load_state_dict(checkpoint['E'])

    if use_cuda:
        extractor.cuda()

    print("Extracting features...")
    src_feat_list = []
    src_gt_list = []
    for idx, (imgs, gt, _) in enumerate(src_val_loader):
        if idx >= th:
            break
        with torch.no_grad():
            if use_cuda:
                imgs, gt = imgs.cuda(), gt.cuda()
            feat = extractor(imgs)
            feat = feat.view(-1)
            src_feat_list.append(feat.cpu().numpy())
            src_gt_list.append(gt.item())

    tgt_feat_list = []
    tgt_gt_list = []
    for idx, (imgs, gt, _) in enumerate(tgt_val_loader):
        if idx >= th:
            break
        with torch.no_grad():
            if use_cuda:
                imgs, gt = imgs.cuda(), gt.cuda()
            feat = extractor(imgs)
            feat = feat.view(-1)
            tgt_feat_list.append(feat.cpu().numpy())
            tgt_gt_list.append(gt.item())

    src_feat_list = np.array(src_feat_list)
    src_gt_list = np.array(src_gt_list)
    tgt_feat_list = np.array(tgt_feat_list)
    tgt_gt_list = np.array(tgt_gt_list)

    all_feat_list = np.concatenate((src_feat_list, tgt_feat_list), axis=0)
    all_gt_list = np.concatenate((src_gt_list, tgt_gt_list), axis=0)

    print("Applying tSNE...")
    all_feat_tsne = TSNE(n_components=2).fit_transform(all_feat_list)

    print("Plotting...")
    plt.figure()
    src_feat_tsne = all_feat_tsne[:src_feat_list.shape[0], :]
    plt.scatter(src_feat_tsne[:, 0], src_feat_tsne[:, 1], c='r', s=5)
    tgt_feat_tsne = all_feat_tsne[src_feat_list.shape[0]:, :]
    plt.scatter(tgt_feat_tsne[:, 0], tgt_feat_tsne[:, 1], c='b', s=5)
    plt.title('{}_tSNE_domain_{}_{}'.format(args.tSNE_model, args.src[0], args.tgt[0]))
    plt.savefig(os.path.join('vis', '{}_tSNE_domain_{}_{}.jpg'.format(args.tSNE_model, args.src[0], args.tgt[0])))
    plt.close()

    plt.figure()
    colors = cm.rainbow(np.linspace(0, 1, 10))
    for i, c in enumerate(colors):
        points = all_feat_tsne[all_gt_list == i]
        plt.scatter(points[:, 0], points[:, 1], c=c, s=5)
    plt.title('{}_tSNE_class_{}_{}'.format(args.tSNE_model, args.src[0], args.tgt[0]))
    plt.savefig(os.path.join('vis', '{}_tSNE_class_{}_{}.jpg'.format(args.tSNE_model, args.src[0], args.tgt[0])))
    plt.close()
