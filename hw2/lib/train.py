import os
import torch
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter

import parser
import models
import data
from trainer import Trainer, evaluate


if __name__ == '__main__':

    args = parser.arg_parse()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # torch.cuda.set_device(args.gpu)
    use_cuda = torch.cuda.is_available()
    
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    print('===> Preparing data...')
    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(data.DATA(args, mode='val'),
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False)

    print('===> Preparing model...')

    model = models.get_model(args)
    if use_cuda:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    writer = SummaryWriter(os.path.join(args.save_dir, 'log_file'))

    my_trainer = Trainer(model=model,
                         train_loader=train_loader,
                         val_loader=val_loader,
                         optim=optimizer,
                         criterion=criterion,
                         args=args,
                         writer=writer,
                         use_cuda=use_cuda
                         )
    print("===> Initializing training...")
    my_trainer.train()

    print("===> Evaluating the best model...")
    best_model = model
    best_model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model_best.pth.tar')))
    if use_cuda:
        best_model = best_model.cuda()
    evaluate(best_model, val_loader, use_cuda=use_cuda)
