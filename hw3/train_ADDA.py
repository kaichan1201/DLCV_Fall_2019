import os
import torch
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter

import lib.parser as parser
import lib.model_ADDA as model_ADDA
import lib.data as data
from lib.trainer_ADDA import Base_Trainer, ADDA_Trainer

if __name__ == '__main__':

    args = parser.arg_parse()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    use_cuda = torch.cuda.is_available()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    print('===> Preparing data...')

    tgt_test_loader = torch.utils.data.DataLoader(data.DigitDataset(args, name=args.tgt, mode='test'),
                                                  batch_size=args.batch_size,
                                                  num_workers=args.workers,
                                                  shuffle=True)

    print('===> Preparing model...')
    writer = SummaryWriter(os.path.join(args.save_dir, 'log'))
    if args.no_transfer:
        train_loader = torch.utils.data.DataLoader(data.DigitDataset(args, name=args.src, mode='train'),
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=True)

        my_model = {
            'E': model_ADDA.E(),
            'C': model_ADDA.C(),
        }
        if use_cuda:
            for _, m in my_model.items():
                m.cuda()
        criterion = nn.CrossEntropyLoss()
        optim = {
            'E': torch.optim.Adam(my_model['E'].parameters(), lr=args.lr, betas=(0.5, 0.999),
                                  weight_decay=args.weight_decay),
            'C': torch.optim.Adam(my_model['C'].parameters(), lr=args.lr, betas=(0.5, 0.999),
                                  weight_decay=args.weight_decay),
        }
        my_trainer = Base_Trainer(model=my_model,
                                  train_loader=train_loader,
                                  val_loader=tgt_test_loader,
                                  optim=optim,
                                  criterion=criterion,
                                  args=args,
                                  writer=writer,
                                  use_cuda=use_cuda,
                                  )
    else:
        train_loader = torch.utils.data.DataLoader(
            data.ConcatDataset(
                data.DigitDataset(args, name=args.src, mode='train'),
                data.DigitDataset(args, name=args.tgt, mode='train'),
            ),
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=True
        )

        src_model = {
            'E': model_ADDA.E(),
            'C': model_ADDA.C(),
        }
        tgt_model = {
            'E': model_ADDA.E(),
            'D': model_ADDA.D(),
        }
        if use_cuda:
            for _, m in src_model.items():
                m.cuda()
            for _, m in tgt_model.items():
                m.cuda()

        criterion = nn.BCELoss()
        optim = {
            'tgt_E': torch.optim.Adam([
                {'params': tgt_model['E'].parameters()},
            ], lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay),
            'D': torch.optim.Adam([
                {'params': tgt_model['D'].parameters()},
            ], lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay),
            'E': None,
            'C': None,
        }
        my_trainer = ADDA_Trainer(src_model=src_model,
                                  tgt_model=tgt_model,
                                  train_loader=train_loader,
                                  val_loader=tgt_test_loader,
                                  optim=optim,
                                  criterion=criterion,
                                  args=args,
                                  writer=writer,
                                  use_cuda=use_cuda,
                                  )

    print("===> Initializing training...")
    my_trainer.train()
