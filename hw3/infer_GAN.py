import os
import numpy as np
import torch
from torchvision.utils import save_image, make_grid

import lib.parser as parser
import lib.model as model


if __name__ == '__main__':

    args = parser.arg_parse()

    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    if args.infer_GAN_mode == 'DC':
        print('inferring DC...')
        # prepare model
        G = model.Generator(nz=args.nz)
        if use_cuda:
            G.cuda()
        G.load_state_dict(torch.load(args.pretrained_G))

        # generate noise
        # noise = torch.randn(32, args.nz, 1, 1)
        noise = torch.tensor(np.loadtxt('noise_DC.txt')).float()
        noise = noise.view(*noise.shape, 1, 1)
        if use_cuda:
            noise = noise.cuda()
        # np.savetxt('noise_DC.txt', noise.squeeze().cpu().numpy())

        # generate images
        imgs = G(noise).detach()
        save_image(imgs, os.path.join(args.save_dir, 'fig1_2.jpg'), nrow=8, padding=2, normalize=True)

    elif args.infer_GAN_mode == 'AC':  # AC
        print('inferring AC...')
        # prepare model
        G = model.AC_Generator(nz=args.nz+1)
        if use_cuda:
            G.cuda()
        G.load_state_dict(torch.load(args.pretrained_G))

        # save 20 images, 10 smiling & 10 not smiling
        # noise = torch.randn(10, args.nz, 1, 1)
        noise = torch.tensor(np.loadtxt('noise_AC.txt')).float()
        noise = noise.view(*noise.shape, 1, 1)
        # np.savetxt('noise_AC.txt', noise.squeeze().cpu().numpy())
        noise = torch.cat((noise, noise), dim=0)
        lbls = torch.cat((torch.zeros(10, 1, 1, 1), torch.ones(10, 1, 1, 1)), dim=0)
        if use_cuda:
            noise = noise.cuda()
            lbls = lbls.cuda()
        imgs = G(noise, lbls)
        save_image(imgs, os.path.join(args.save_dir, 'fig2_2.jpg'), nrow=10, padding=2, normalize=True)
