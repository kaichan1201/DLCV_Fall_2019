import os
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from mean_iou_evaluate import mean_iou_score


def evaluate(model, data_loader, use_cuda):
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():  # do not need to calculate information for gradient during eval
        for idx, (imgs, gt) in enumerate(data_loader):
            if use_cuda:
                imgs = imgs.cuda()
            out = model(imgs)
            _, pred = torch.max(out, dim=1)

            pred = pred.cpu().numpy().squeeze()
            gt = gt.numpy().squeeze()

            preds.append(pred)
            gts.append(gt)
    gts = np.concatenate(gts)
    preds = np.concatenate(preds)

    return mean_iou_score(preds, gts)


class Trainer:
    def __init__(self, model, train_loader, val_loader, optim, criterion, args, writer, use_cuda):
        self.model = model
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
            print("===>Loading pretrained model {}...".format(self.args.pretrained))
            self.model.load_state_dict(torch.load(self.args.pretrained))

    def get_lr(self):
        return self.base_lr * (1 - self.iters/self.max_iter)**0.9

    def train(self):
        self.model.train()
        for epoch in range(1, self.epochs+1):
            self.total_cnt, self.correct_cnt = 0, 0
            current_iter = 0

            '''set new lr'''
            for param in self.optim.param_groups:
                print('Epoch {}, New lr: {}'.format(epoch, self.get_lr()))
                param['lr'] = self.get_lr()

            '''train an epoch'''
            for idx, (imgs, lbls) in enumerate(self.train_loader):
                self.iters += 1
                current_iter += 1
                if self.use_cuda:
                    imgs, lbls = imgs.cuda(), lbls.cuda()

                '''model forwarding & loss calculation'''
                out = self.model(imgs)
                loss = self.criterion(out, lbls)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                with torch.no_grad():
                    _, preds = torch.max(out, dim=1)
                    self.total_cnt += preds.cpu().numpy().size
                    self.correct_cnt += (preds == lbls).sum().item()

                self.writer.add_scalar('loss', loss.item(), self.iters)
                print('Epoch [{}][{}/{}], Loss: {:4f}'.format(epoch, current_iter,
                                                              len(self.train_loader), loss.item()))

            train_acc = self.correct_cnt / self.total_cnt
            self.writer.add_scalar('train_acc', train_acc, epoch)
            print('Epoch {}, Train Acc: {:4f}'.format(epoch, train_acc))

            if epoch % self.val_epoch == 0:
                val_acc = evaluate(self.model, self.val_loader, use_cuda=self.use_cuda)
                self.writer.add_scalar('val_iou', val_acc, epoch)

                if val_acc > self.best_acc:
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_best.pth.tar'))
                    self.best_acc = val_acc

            if epoch % self.save_epoch == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_{}.pth.tar'.format(epoch)))
