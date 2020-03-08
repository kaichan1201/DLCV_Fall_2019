import os
import torch
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter

import lib.parser as parser
import lib.model as model
import lib.data as data
from lib.trainer_no_RNN import Trainer

if __name__ == '__main__':

    args = parser.arg_parse()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    use_cuda = torch.cuda.is_available()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    print('===> Preparing data...')

    train_loader = torch.utils.data.DataLoader(data.TrimmedVideoDataset(args, mode='train', model_type='CNN'),
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,)

    val_loader = torch.utils.data.DataLoader(data.TrimmedVideoDataset(args, mode='valid', model_type='CNN'),
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=True)

    writer = SummaryWriter(os.path.join(args.save_dir, 'log'))
    print('===> Preparing model...')
    my_model = {
        'E': model.E_Resnet50(pretrained=True),
        'C': model.C(in_size=args.frame_num * 2048),
    }
    if use_cuda:
        for _, m in my_model.items():
            m.cuda()

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam([
        {'params': my_model['C'].parameters()},
    ], lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)

    my_trainer = Trainer(model=my_model,
                         train_loader=train_loader,
                         val_loader=val_loader,
                         optim=optim,
                         criterion=criterion,
                         args=args,
                         writer=writer,
                         use_cuda=use_cuda
                         )

    print("===> Initializing training...")
    my_trainer.train()
