import os
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)


class Base_Trainer:
    def __init__(self, model, train_loader, val_loader, optim, criterion, args, writer, use_cuda):
        assert isinstance(model, dict)
        self.E = model['E']
        self.C = model['C']
        self.train_loader = train_loader
        self.val_loader = val_loader
        assert isinstance(optim, dict)
        self.optim_E = optim['E']
        self.optim_C = optim['C']
        self.criterion = criterion
        self.args = args
        self.writer = writer
        self.use_cuda = use_cuda

        self.epochs = self.args.epochs
        self.val_epoch = self.args.val_epoch
        self.save_epoch = self.args.save_epoch
        self.save_dir = self.args.save_dir
        self.base_lr = self.args.lr

        self.iters = 0
        self.max_iter = len(self.train_loader) * self.epochs
        self.best_acc = 0
        self.total_cnt = 0
        self.correct_cnt = 0

        if len(self.args.pretrained):
            print("===>Loading pretrained model {}...".format(self.args.pretrained))
            checkpoint = torch.load(self.args.pretrained)
            self.E.load_state_dict(checkpoint['E'])
            self.C.load_state_dict(checkpoint['C'])
        else:
            self.E.apply(weights_init)
            self.C.apply(weights_init)

    def get_lr(self):
        return self.base_lr * (1 - self.iters / self.max_iter) ** 0.9

    def train(self):
        self.E.train()
        self.C.train()
        for epoch in range(1, self.epochs + 1):
            self.total_cnt, self.correct_cnt = 0, 0
            current_iter = 0

            '''set new lr'''
            print('Epoch {}, New lr: {}'.format(epoch, self.get_lr()))
            for param in self.optim_E.param_groups:
                param['lr'] = self.get_lr()
            for param in self.optim_C.param_groups:
                param['lr'] = self.get_lr()

            '''train an epoch'''
            for idx, (imgs, lbls) in enumerate(self.train_loader):
                self.iters += 1
                current_iter += 1
                if self.use_cuda:
                    imgs, lbls = imgs.cuda(), lbls.cuda()

                '''model forwarding & loss calculation'''
                out = self.C(self.E(imgs))
                loss = self.criterion(out, lbls)

                self.optim_C.zero_grad()
                self.optim_E.zero_grad()
                loss.backward()
                self.optim_C.step()
                self.optim_E.step()

                with torch.no_grad():
                    _, preds = torch.max(out, dim=1)
                    self.total_cnt += preds.cpu().numpy().size
                    self.correct_cnt += (preds == lbls).sum().item()

                self.writer.add_scalar('loss', loss.item(), self.iters)
                if current_iter % 50 == 0 or current_iter == len(self.train_loader):
                    print('Epoch [{}][{}/{}], Loss: {:.4f}'.format(epoch, current_iter,
                                                                   len(self.train_loader), loss.item()))

            train_acc = self.correct_cnt / self.total_cnt
            self.writer.add_scalar('acc/train_acc', train_acc, epoch)
            print('Epoch {}, Train Acc: {:.4f}'.format(epoch, train_acc))

            if epoch % self.val_epoch == 0:
                val_acc = self.evaluate()
                self.writer.add_scalar('acc/val_acc', val_acc, epoch)
                print('Epoch {}, Val Acc: {:.4f}'.format(epoch, val_acc))

                if val_acc > self.best_acc:
                    torch.save(
                        {
                            'E': self.E.state_dict(),
                            'C': self.C.state_dict(),
                        },
                        os.path.join(self.save_dir, 'checkpoint_best.pth.tar')
                    )
                    self.best_acc = val_acc

            if epoch % self.save_epoch == 0:
                torch.save(
                    {
                        'E': self.E.state_dict(),
                        'C': self.C.state_dict(),
                    },
                    os.path.join(self.save_dir, 'checkpoint_{}.pth.tar'.format(epoch))
                )
        print('\nBest Val Acc: {:.6f}\n'.format(self.best_acc))

    def evaluate(self):
        self.E.eval()
        self.C.eval()
        total_cnt = 0
        correct_cnt = 0
        with torch.no_grad():  # do not need to calculate information for gradient during eval
            for idx, (imgs, gt, _) in enumerate(self.val_loader):
                if self.use_cuda:
                    imgs, gt = imgs.cuda(), gt.cuda()
                out = self.C(self.E(imgs))
                _, pred = torch.max(out, dim=1)

                total_cnt += pred.cpu().numpy().size
                correct_cnt += (pred == gt).sum().item()

        return correct_cnt / total_cnt


class ADDA_Trainer(Base_Trainer):
    def __init__(self, src_model, tgt_model, train_loader, val_loader, optim, criterion, args, writer, use_cuda):
        super(ADDA_Trainer, self).__init__(src_model, train_loader, val_loader, optim, criterion, args, writer, use_cuda)
        assert isinstance(src_model, dict)
        self.src_E = src_model['E']
        self.src_C = src_model['C']
        assert len(self.args.pretrained_src)
        pretrained_src = torch.load(self.args.pretrained_src)
        self.src_E.load_state_dict(pretrained_src['E'])
        self.src_C.load_state_dict(pretrained_src['C'])

        assert isinstance(tgt_model, dict)
        self.tgt_E = tgt_model['E']
        self.D = tgt_model['D']

        if self.args.pretrained:
            checkpoint = torch.load(self.args.pretrained)
            self.tgt_E.load_state_dict(checkpoint['E'])
            self.D.load_state_dict(checkpoint['D'])
        else:
            self.tgt_E.load_state_dict(pretrained_src['E'])
            self.D.apply(weights_init)

        self.criterion_bce = criterion

        self.optim_tgt_E = optim['tgt_E']
        self.optim_D = optim['D']

        self.flip_y_freq = self.args.flip_y_freq
        self.noisy_std = self.args.noisy_input

    # def get_lr(self):
    #     return self.base_lr

    def get_flip_y_freq(self):
        return self.flip_y_freq * (1 - self.iters / self.max_iter) ** 0.9

    def get_std(self):
        return self.noisy_std * (1 - self.iters / self.max_iter) ** 0.9

    def train(self):
        # self.evaluate(0)
        for epoch in range(1, self.epochs + 1):
            self.src_E.eval()
            self.tgt_E.train()
            self.D.train()

            current_iter = 0

            '''set new lr'''
            print('\nEpoch {}, New lr: {}'.format(epoch, self.get_lr()))
            for param in self.optim_tgt_E.param_groups:
                param['lr'] = self.get_lr()
            for param in self.optim_D.param_groups:
                param['lr'] = self.get_lr()

            '''train an epoch'''
            for idx, ((s_imgs, s_lbls), (t_imgs, t_lbls)) in enumerate(self.train_loader):
                self.iters += 1
                current_iter += 1

                if self.noisy_std:
                    t_imgs = t_imgs + torch.normal(mean=torch.zeros_like(t_imgs),
                                                   std=torch.ones_like(t_imgs)*self.get_std())

                '''calculate y'''
                if self.args.soft_label:  # soft label
                    y_real = 0.7 + (1.2 - 0.7) * torch.rand(s_imgs.shape[0])
                    y_fake = 0.0 + (0.3 - 0.0) * torch.rand(s_imgs.shape[0])
                else:  # hard label
                    y_real = torch.ones(s_imgs.shape[0])
                    y_fake = torch.zeros(s_imgs.shape[0])

                # add noise labels
                y_noise_idx = torch.rand(s_imgs.shape[0]) < self.get_flip_y_freq()
                y_real.masked_fill_(y_noise_idx, 0)
                y_fake.masked_fill_(y_noise_idx, 1)

                if self.use_cuda:
                    s_imgs, s_lbls = s_imgs.cuda(), s_lbls.cuda()
                    t_imgs, t_lbls = t_imgs.cuda(), t_lbls.cuda()
                    y_real, y_fake = y_real.cuda(), y_fake.cuda()

                '''model forwarding & loss calculation'''
                # update D
                t_emb, D_x, D_gz1, D_loss = self.update_D(s_imgs, t_imgs, y_real, y_fake)
                # update G
                D_gz2, G_loss = self.update_tgt_E(t_emb, y_real)

                self.writer.add_scalar('loss/G_loss', G_loss.item(), self.iters)
                self.writer.add_scalar('loss/D_loss', D_loss.item(), self.iters)
                self.writer.add_scalar('pred/D(x)', D_x, self.iters)
                self.writer.add_scalar('pred/D(G(z))', D_gz1, self.iters)

                if current_iter % 50 == 0 or current_iter == len(self.train_loader):
                    print('Epoch [{}][{}/{}] G_loss:{:.4f} | D_loss:{:.4f} | D(x):{:.4f} | D(G(z)):{:.4f}/{:.4f}'
                          .format(epoch, current_iter, len(self.train_loader), G_loss.item(), D_loss.item(),
                                  D_x, D_gz1, D_gz2))

            if epoch % self.val_epoch == 0:
                self.evaluate(epoch)

            if epoch % self.save_epoch == 0:
                torch.save(
                    {
                        'D': self.D.state_dict(),
                        'E': self.tgt_E.state_dict(),
                    },
                    os.path.join(self.save_dir, 'checkpoint_{}.pth.tar'.format(epoch))
                )
        print('\nBest Val Acc: {:.6f}\n'.format(self.best_acc))

    def update_D(self, s_imgs, t_imgs, y_real, y_fake):
        self.D.zero_grad()
        # src (real)
        real_adv_out = self.D(self.src_E(s_imgs).detach())
        real_adv_loss = self.criterion_bce(real_adv_out, y_real)
        real_adv_loss.backward()
        # print(real_adv_out)
        D_x = real_adv_out.mean().item()

        # tgt (fake)
        t_emb = self.tgt_E(t_imgs)
        fake_adv_out = self.D(t_emb.detach())
        fake_adv_loss = self.criterion_bce(fake_adv_out, y_fake)
        fake_adv_loss.backward()
        D_gz1 = fake_adv_out.mean().item()

        D_loss = real_adv_loss + fake_adv_loss
        self.optim_D.step()

        return t_emb, D_x, D_gz1, D_loss

    def update_tgt_E(self, t_emb, y_real):
        self.optim_tgt_E.zero_grad()

        adv_out = self.D(t_emb)
        G_loss = self.criterion_bce(adv_out, y_real)
        G_loss.backward()
        D_gz2 = adv_out.mean().item()

        self.optim_tgt_E.step()
        return D_gz2, G_loss

    def evaluate(self, epoch):
        self.D.eval()
        self.tgt_E.eval()
        self.C.eval()

        total_cnt, total_correct_cnt = 0, 0

        with torch.no_grad():
            for idx, (imgs, gts, _) in enumerate(self.val_loader):
                if self.use_cuda:
                    imgs, gts = imgs.cuda(), gts.cuda()
                out = self.C(self.tgt_E(imgs))
                _, pred = torch.max(out, dim=1)
                total_cnt += pred.cpu().numpy().size
                total_correct_cnt += (pred == gts).sum().item()

        val_acc = total_correct_cnt / total_cnt
        self.writer.add_scalar('acc/val_acc', val_acc, epoch)
        print('Epoch {}, Tgt Val Acc: {:.6f}'.format(epoch, val_acc))

        if val_acc > self.best_acc:
            torch.save(
                {
                    'D': self.D.state_dict(),
                    'E': self.tgt_E.state_dict(),
                },
                os.path.join(self.save_dir, 'checkpoint_best.pth.tar')
            )
            self.best_acc = val_acc
