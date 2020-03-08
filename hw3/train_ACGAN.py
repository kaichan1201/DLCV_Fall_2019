import os
import torch
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter

import lib.parser as parser
import lib.model as model
import lib.data as data
from lib.trainer_GAN import ACGAN_Trainer

if __name__ == '__main__':

    args = parser.arg_parse()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    use_cuda = torch.cuda.is_available()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    print('===> Preparing data...')
    train_loader = torch.utils.data.DataLoader(data.FaceDataset(args, mode='train'),
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True)

    print('===> Preparing model...')
    G = model.AC_Generator(nz=args.nz+1)
    D = model.AC_Discriminator()
    if use_cuda:
        G.cuda()
        D.cuda()

    criterion = {
        'bce': nn.BCELoss(),
        'cls': nn.CrossEntropyLoss(),
    }
    optim_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    optim_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    # optim_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # optim_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    writer = SummaryWriter(os.path.join(args.save_dir, 'log'))

    my_trainer = ACGAN_Trainer(G=G,
                               D=D,
                               train_loader=train_loader,
                               optim_G=optim_G,
                               optim_D=optim_D,
                               criterion=criterion,
                               args=args,
                               writer=writer,
                               use_cuda=use_cuda
                               )
    print("===> Initializing training...")
    my_trainer.train()
