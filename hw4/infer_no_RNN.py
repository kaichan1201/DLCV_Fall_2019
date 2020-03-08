import os
import torch
import numpy as np
import pandas as pd

import lib.parser as parser
import lib.model as model
import lib.data as data

if __name__ == '__main__':

    args = parser.arg_parse()

    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    # prepare data
    val_loader = torch.utils.data.DataLoader(data.TrimmedVideoDataset(args, mode='valid', model_type='CNN'),
                                             batch_size=1,
                                             num_workers=args.workers,
                                             shuffle=False)

    # prepare model
    my_model = {
        'E': model.E_Resnet50(pretrained=True),
        'C': model.C(in_size=args.frame_num * 2048),
    }

    if use_cuda:
        for _, m in my_model.items():
            m.cuda()
    checkpoint = torch.load(args.pretrained)
    my_model['C'].load_state_dict(checkpoint['C'])

    pred_list = []

    my_model['E'].eval()
    my_model['C'].eval()
    with torch.no_grad():  # do not need to calculate information for gradient during eval
        for idx, data in enumerate(val_loader):

            videos = data[0]
            if use_cuda:
                videos = videos.cuda()

            feats = []
            for i in range(args.frame_num):
                feat = my_model['E'](videos[:, i, :, :, :])
                feats.append(feat)
            feats = torch.cat(feats, dim=1)

            out = my_model['C'](feats)
            _, pred = torch.max(out, dim=1)
            pred_list.append(pred.item())

        with open(os.path.join(args.save_dir, 'p1_valid.txt'), 'w') as f:
            for pred in pred_list:
                f.write('{}\n'.format(pred))
