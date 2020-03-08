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
    val_loader = torch.utils.data.DataLoader(data.TrimmedVideoDataset(args, mode='test', model_type='RNN'),
                                             batch_size=1,
                                             num_workers=args.workers,
                                             shuffle=False)

    # prepare model
    my_model = {
        'E': model.E_Resnet50(pretrained=True),
        'RNN': model.RNN(),
    }

    if use_cuda:
        for _, m in my_model.items():
            m.cuda()
    checkpoint = torch.load(args.pretrained)
    my_model['RNN'].load_state_dict(checkpoint['RNN'])

    pred_list = []
    total_cnt = 0
    correct_cnt = 0

    my_model['E'].eval()
    my_model['RNN'].eval()
    with torch.no_grad():  # do not need to calculate information for gradient during eval
        for idx, data in enumerate(val_loader):
            video = data[0]
            if use_cuda:
                video = video.cuda()  # video: (1, T, C, H, W)
            video = video.squeeze(0)  # video: (T, C, H, W)

            '''extracting features'''
            feat = my_model['E'](video)
            feats = [feat]

            '''model forwarding & loss calculation'''
            out = my_model['RNN'](feats)
            _, pred = torch.max(out, dim=1)
            pred_list.append(pred.item())

        with open(os.path.join(args.save_dir, 'p2_result.txt'), 'w') as f:
            for pred in pred_list:
                f.write('{}\n'.format(pred))
