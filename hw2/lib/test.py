import os
import torch

import parser
import models
import data
from trainer import evaluate


if __name__ == '__main__':
    
    args = parser.arg_parse()

    # torch.cuda.set_device(args.gpu)
    use_cuda = torch.cuda.is_available()

    print('===> Preparing data...')
    test_loader = torch.utils.data.DataLoader(data.DATA(args, mode='val'),
                                              batch_size=args.batch_size,
                                              num_workers=args.workers,
                                              shuffle=False)

    print('===> Preparing model...')
    model = models.get_model(args)
    if use_cuda:
        model = model.cuda()

    checkpoint = torch.load(args.pretrained)
    model.load_state_dict(checkpoint)

    print('===> Evaluating...')
    acc = evaluate(model, test_loader, use_cuda=use_cuda)
    print('Val mIoU: {}'.format(acc))
