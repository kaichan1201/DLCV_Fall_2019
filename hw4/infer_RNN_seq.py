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
    val_loader = torch.utils.data.DataLoader(data.LongVideoDataset_Infer(args, mode='test'),
                                             batch_size=1,
                                             num_workers=args.workers,
                                             shuffle=False)

    # prepare model
    my_model = {
        'E': model.E_Resnet50(pretrained=True),
        'RNN': model.RNN_Seq(),
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
    gru = my_model['RNN'].gru
    classifier = my_model['RNN'].classifier

    with torch.no_grad():  # do not need to calculate information for gradient during eval
        for idx, (video, name) in enumerate(val_loader):
            h = torch.zeros(2, video.shape[0], 512)
            if use_cuda:
                video = video.cuda()  # video: (1, T, C, H, W)
                h = h.cuda()
            name = name[0]
            print('Processing {}...'.format(name))

            for t in range(video.shape[1]):
                frame = video[:, t, :, :, :]
                '''extracting features'''
                feat = my_model['E'](frame)

                '''model forwarding & loss calculation'''
                out, h = gru(feat.unsqueeze(1), h)
                _, pred = torch.max(classifier(out[:, -1, :]), dim=1)
                pred_list.append(pred.item())

            with open(os.path.join(args.save_dir, '{}.txt'.format(name)), 'w') as f:
                for pred in pred_list:
                    f.write('{}\n'.format(pred))

            pred_list.clear()
            torch.cuda.empty_cache()
