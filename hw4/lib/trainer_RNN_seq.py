import os
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.utils.rnn import pad_sequence


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Trainer:
    def __init__(self, model, train_loader, val_loader, optim, criterion, args, writer, use_cuda):
        assert(isinstance(model, dict))
        self.E = model['E']
        self.RNN = model['RNN']

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optim = optim
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
            print("===> Loading pretrained model {}...".format(self.args.pretrained))
            checkpoint = torch.load(self.args.pretrained)
            self.RNN.load_state_dict(checkpoint['RNN'])

    def get_lr(self):
        return self.base_lr * (1 - self.iters / self.max_iter) ** 0.9

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.E.eval()
            self.RNN.train()

            self.total_cnt, self.correct_cnt = 0, 0
            current_iter = 0

            '''set new lr'''
            for param in self.optim.param_groups:
                print('\nEpoch {}, New lr: {}'.format(epoch, self.get_lr()))
                param['lr'] = self.get_lr()

            '''train an epoch'''
            for idx, (video, lbl) in enumerate(self.train_loader):
                self.iters += 1
                current_iter += 1
                if self.use_cuda:
                    video, lbl = video.cuda(), lbl.cuda()  # video: (1, T, C, H, W)
                video = video.squeeze(0)  # video: (T, C, H, W)
                lbl = lbl.squeeze()  # lbl: (T,)

                '''extracting features'''
                with torch.no_grad():
                    feat = self.E(video)  # (T, feat_size)

                '''model forwarding & loss calculation'''
                out = self.RNN(feat)  # (T, n_classes)
                loss = self.criterion(out, lbl)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                with torch.no_grad():
                    _, preds = torch.max(out, dim=1)
                    self.total_cnt += preds.cpu().numpy().size
                    self.correct_cnt += (preds == lbl).sum().item()

                self.writer.add_scalar('loss', loss.item(), self.iters)
                if current_iter % 25 == 0 or current_iter == len(self.train_loader):
                    print('Epoch [{}][{}/{}], Loss: {:.4f}'.format(epoch, current_iter, len(self.train_loader),
                                                                   loss.item()))
                '''reset cache'''
                torch.cuda.empty_cache()

            train_acc = self.correct_cnt / self.total_cnt
            self.writer.add_scalar('acc/train_acc', train_acc, epoch)
            print('Epoch {}, Train Acc: {:.4f}'.format(epoch, train_acc))

            if epoch % self.val_epoch == 0:
                self.evaluate(epoch)

            if epoch % self.save_epoch == 0:
                torch.save({
                    'RNN': self.RNN.state_dict(),
                }, os.path.join(self.save_dir, 'checkpoint_{}.pth.tar'.format(epoch)))

        print('Best Acc: {:.4f}'.format(self.best_acc))

    def evaluate(self, epoch):
        self.E.eval()
        self.RNN.eval()
        total_cnt = 0
        correct_cnt = 0
        with torch.no_grad():  # do not need to calculate information for gradient during eval
            for idx, (video, gt) in enumerate(self.val_loader):
                if self.use_cuda:
                    video, gt = video.cuda(), gt.cuda()  # video: (1, T, C, H, W)
                video = video.squeeze(0)
                gt = gt.squeeze()

                '''extracting features'''
                feat = self.E(video)
                out = self.RNN(feat)
                _, pred = torch.max(out, dim=1)

                total_cnt += pred.cpu().numpy().size
                correct_cnt += (pred == gt).sum().item()

        val_acc = correct_cnt / total_cnt
        self.writer.add_scalar('acc/val_acc', val_acc, epoch)
        print('Epoch {}, Val Acc: {:.4f}'.format(epoch, val_acc))

        if val_acc > self.best_acc:
            torch.save({
                'RNN': self.RNN.state_dict(),
            }, os.path.join(self.save_dir, 'checkpoint_best.pth.tar'))
            self.best_acc = val_acc
