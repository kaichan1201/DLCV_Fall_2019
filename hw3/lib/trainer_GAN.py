import os
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGAN_Trainer:
    def __init__(self, G, D, train_loader, optim_G, optim_D, criterion, args, writer, use_cuda):
        self.G = G
        self.D = D
        self.train_loader = train_loader
        self.optim_G = optim_G
        self.optim_D = optim_D
        self.criterion = criterion
        self.args = args
        self.writer = writer
        self.use_cuda = use_cuda

        self.epochs = self.args.epochs
        self.val_epoch = self.args.val_epoch
        self.save_epoch = self.args.save_epoch
        self.save_dir = self.args.save_dir
        self.base_lr = self.args.lr
        self.nz = self.args.nz
        self.flip_y_freq = self.args.flip_y_freq

        self.iters = 0
        self.max_iter = len(self.train_loader) * self.epochs

        if self.args.pretrained_G:
            print('Using pretrained G...')
            self.G.load_state_dict(torch.load(self.args.pretrained_G))
        else:
            self.G.apply(weights_init)
        if self.args.pretrained_D:
            print('Using pretrained D...')
            self.D.load_state_dict(torch.load(self.args.pretrained_D))
        else:
            self.D.apply(weights_init)
        if self.args.soft_label:
            print('Using soft label...')

    def get_lr(self):
        return self.base_lr * (1 - self.iters / self.max_iter) ** 0.9
        # return self.base_lr

    def get_flip_y_freq(self):
        return self.flip_y_freq * (1 - self.iters / self.max_iter) ** 0.9

    def train(self):
        self.G.train()
        self.D.train()
        for epoch in range(1, self.epochs + 1):
            current_iter = 0

            '''set new lr'''
            print('Epoch {}, New lr: {:.4f}, New flip freq: {:.4f}'
                  .format(epoch, self.get_lr(), self.get_flip_y_freq()))
            for param in self.optim_G.param_groups:
                param['lr'] = self.get_lr()
            for param in self.optim_D.param_groups:
                param['lr'] = self.get_lr()

            '''train an epoch'''
            for idx, (imgs, lbls) in enumerate(self.train_loader):
                self.iters += 1
                current_iter += 1

                if self.args.soft_label:  # soft label
                    y_real = 0.7 + (1.2 - 0.7)*torch.rand(imgs.shape[0])
                    y_fake = 0.0 + (0.3 - 0.0)*torch.rand(imgs.shape[0])
                else:  # hard label
                    y_real = torch.ones(imgs.shape[0])
                    y_fake = torch.zeros(imgs.shape[0])

                # add noise labels
                y_noise_idx = torch.rand(imgs.shape[0]) < self.get_flip_y_freq()
                y_real.masked_fill_(y_noise_idx, 0)
                y_fake.masked_fill_(y_noise_idx, 1)

                noise = torch.randn(imgs.shape[0], self.nz, 1, 1)
                fake_lbls = torch.randint(0, 1+1, (imgs.shape[0],))
                if self.use_cuda:
                    imgs, lbls = imgs.cuda(), lbls.cuda()
                    noise, fake_lbls = noise.cuda(), fake_lbls.cuda()
                    y_real, y_fake = y_real.cuda(), y_fake.cuda()

                '''update D'''
                fake_imgs, D_x, D_gz1, D_loss = self.update_D(imgs, lbls, noise, fake_lbls, y_real, y_fake)

                '''update G'''
                D_gz2, G_loss = self.update_G(fake_imgs, fake_lbls, y_real)

                self.writer.add_scalar('loss/G_loss', G_loss.item(), self.iters)
                self.writer.add_scalar('loss/D_loss', D_loss.item(), self.iters)
                self.writer.add_scalar('pred/D(x)', D_x, self.iters)
                self.writer.add_scalar('pred/D(G(z))', D_gz1, self.iters)

                if current_iter % 1 == 0 or current_iter == len(self.train_loader):
                    print('Epoch [{}][{}/{}] G_loss:{:.4f} | D_loss:{:.4f} | D(x):{:.4f} | D(G(z)):{:.4f}/{:.4f}'
                          .format(epoch, current_iter, len(self.train_loader), G_loss.item(), D_loss.item(),
                                  D_x, D_gz1, D_gz2))

            if epoch % self.val_epoch == 0:
                self.validate()

            if epoch % self.save_epoch == 0:
                # save 8 images
                self.save_vis(epoch)
                # save model
                torch.save(self.G.state_dict(), os.path.join(self.save_dir, 'G_{}.pth.tar'.format(epoch)))
                torch.save(self.D.state_dict(), os.path.join(self.save_dir, 'D_{}.pth.tar'.format(epoch)))

    def validate(self):
        pass  # do nothing for GAN

    def update_D(self, imgs, lbls, noise, fake_lbls, y_real, y_fake):
        self.optim_D.zero_grad()

        # real img
        out_real = self.D(imgs)
        real_D_loss = self.criterion(out_real, y_real)
        real_D_loss.backward()
        D_x = out_real.mean().item()

        # fake img
        fake_imgs = self.G(noise)
        out_fake = self.D(fake_imgs.detach())
        fake_D_loss = self.criterion(out_fake, y_fake)
        fake_D_loss.backward()
        D_gz1 = out_fake.mean().item()

        self.optim_D.step()
        D_loss = real_D_loss + fake_D_loss
        return fake_imgs, D_x, D_gz1, D_loss

    def update_G(self, fake_imgs, fake_lbls, y_real):  # fake_imgs are generated in update_D (by G)
        self.optim_G.zero_grad()
        out = self.D(fake_imgs)
        G_loss = self.criterion(out, y_real)
        G_loss.backward()
        D_gz2 = out.mean().item()
        self.optim_G.step()
        return D_gz2, G_loss

    def save_vis(self, epoch):
        # save 10 images
        noise = torch.randn(8, self.nz, 1, 1)
        if self.use_cuda:
            noise = noise.cuda()
        imgs = self.G(noise).detach()
        self.writer.add_image('vis', make_grid(imgs, nrow=5, padding=10, normalize=True), epoch)


class ACGAN_Trainer(DCGAN_Trainer):
    def __init__(self, G, D, train_loader, optim_G, optim_D, criterion, args, writer, use_cuda):
        super(ACGAN_Trainer, self).__init__(G, D, train_loader, optim_G, optim_D, criterion, args, writer, use_cuda)
        assert isinstance(criterion, dict)
        self.criterion_bce = criterion['bce']
        self.criterion_cls = criterion['cls']

    def update_D(self, imgs, lbls, noise, fake_lbls, y_real, y_fake):
        self.optim_D.zero_grad()

        # real img
        out_real, pred_real = self.D(imgs)
        real_D_adv_loss = self.criterion_bce(out_real, y_real)
        real_D_cls_loss = self.criterion_cls(pred_real, lbls)

        real_D_loss = (real_D_adv_loss + real_D_cls_loss) / 2
        real_D_loss.backward()
        D_x = out_real.mean().item()

        # fake img
        fake_imgs = self.G(noise, fake_lbls)
        out_fake, pred_fake = self.D(fake_imgs.detach())
        fake_D_adv_loss = self.criterion_bce(out_fake, y_fake)
        fake_D_cls_loss = self.criterion_cls(pred_fake, fake_lbls)

        fake_D_loss = (fake_D_adv_loss + fake_D_cls_loss) / 2
        fake_D_loss.backward()
        D_gz1 = out_fake.mean().item()

        self.optim_D.step()
        D_loss = real_D_loss + fake_D_loss
        return fake_imgs, D_x, D_gz1, D_loss

    def update_G(self, fake_imgs, fake_lbls, y_real):  # fake_imgs are generated in update_D (by G)
        self.optim_G.zero_grad()
        out, pred = self.D(fake_imgs)
        G_adv_loss = self.criterion_bce(out, y_real)
        G_cls_loss = self.criterion_cls(pred, fake_lbls)

        G_loss = (G_adv_loss + G_cls_loss) / 2
        G_loss.backward()
        D_gz2 = out.mean().item()

        self.optim_G.step()
        return D_gz2, G_loss

    def save_vis(self, epoch):
        # save 8 images, 4 smiling & 4 not smiling
        noise = torch.randn(5, self.nz, 1, 1)
        noise = torch.cat((noise, noise), dim=0)
        lbls = torch.cat((torch.zeros(5, 1, 1, 1), torch.ones(5, 1, 1, 1)), dim=0)
        if self.use_cuda:
            noise = noise.cuda()
            lbls = lbls.cuda()
        imgs = self.G(noise, lbls).detach()
        self.writer.add_image('vis', make_grid(imgs, nrow=5, padding=10, normalize=True), epoch)
