import os
import cv2
import torch

import parser
import models
import data

if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()

    args = parser.arg_parse()

    print('===> Preparing data...')  # no seg!
    infer_loader = torch.utils.data.DataLoader(data.INFER_DATA(args),
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=False)

    print('===> Preparing model...')
    model = models.get_model(args)

    if use_cuda:
        model = model.cuda()

    checkpoint = torch.load(args.pretrained)
    model.load_state_dict(checkpoint)

    print('===> Inferring...')
    model.eval()
    for _, (imgs, img_names) in enumerate(infer_loader):
        if use_cuda:
            imgs = imgs.cuda()

        with torch.no_grad():
            out = model(imgs)
            _, pred = torch.max(out, dim=1)

        for idx in range(imgs.shape[0]):
            cv2.imwrite(os.path.join(args.save_dir, img_names[idx]), pred[idx, :, :].cpu().numpy().squeeze())
        # print('Saving {}...'.format(os.path.join(args.save_dir, img_names)))
