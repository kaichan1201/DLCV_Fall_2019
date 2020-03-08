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


def evaluate(model, data_loader, use_cuda):
    model.eval()
    total_cnt = 0
    correct_cnt = 0
    with torch.no_grad():  # do not need to calculate information for gradient during eval
        for idx, (imgs, gt, _) in enumerate(data_loader):
            if use_cuda:
                imgs, gt = imgs.cuda(), gt.cuda()
            out = model(imgs)
            _, pred = torch.max(out, dim=1)

            total_cnt += pred.cpu().numpy().size
            correct_cnt += (pred == gt).sum().item()

    return correct_cnt / total_cnt


class Base_Trainer:
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
        return self.base_lr * (1 - self.iters / self.max_iter) ** 0.9

    def train(self):
        self.model.train()
        for epoch in range(1, self.epochs + 1):
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
                if current_iter % 50 == 0 or current_iter == len(self.train_loader):
                    print('Epoch [{}][{}/{}], Loss: {:.4f}'.format(epoch, current_iter,
                                                                   len(self.train_loader), loss.item()))

            train_acc = self.correct_cnt / self.total_cnt
            self.writer.add_scalar('acc/train_acc', train_acc, epoch)
            print('Epoch {}, Train Acc: {:.4f}'.format(epoch, train_acc))

            if epoch % self.val_epoch == 0:
                val_acc = evaluate(self.model, self.val_loader, use_cuda=self.use_cuda)
                self.writer.add_scalar('acc/val_acc', val_acc, epoch)
                print('Epoch {}, Val Acc: {:.4f}'.format(epoch, val_acc))

                if val_acc > self.best_acc:
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_best.pth.tar'))
                    self.best_acc = val_acc

            if epoch % self.save_epoch == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_{}.pth.tar'.format(epoch)))


class DANN_Trainer(Base_Trainer):
    def __init__(self, model, train_loader, val_loader, optim, criterion, args, writer, use_cuda):
        super(DANN_Trainer, self).__init__(model, train_loader, val_loader, optim, criterion, args, writer, use_cuda)
        assert isinstance(criterion, dict)
        self.criterion_bce = criterion['bce']
        self.criterion_cls = criterion['cls']

        self.flip_y_freq = self.args.flip_y_freq

    def get_lr(self):
        return self.base_lr

    def get_flip_y_freq(self):
        return self.flip_y_freq * (1 - self.iters / self.max_iter) ** 0.9

    def get_alpha(self):
        return 2 / (1 + np.exp(-10 * self.iters / self.max_iter)) - 1

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.total_cnt, self.correct_cnt = 0, 0
            current_iter = 0

            '''set new lr'''
            for param in self.optim.param_groups:
                print('\nEpoch {}, New lr: {}'.format(epoch, self.get_lr()))
                param['lr'] = self.get_lr()

            print('\nEpoch {}, New alpha: {}'.format(epoch, self.get_alpha()))

            '''train an epoch'''
            for idx, ((imgs, lbls), (t_imgs, t_lbls)) in enumerate(self.train_loader):
                self.iters += 1
                current_iter += 1
                self.optim.zero_grad()

                '''calculate y'''
                if self.args.soft_label:  # soft label
                    y_real = 0.7 + (1.2 - 0.7) * torch.rand(imgs.shape[0])
                    y_fake = 0.0 + (0.3 - 0.0) * torch.rand(imgs.shape[0])
                else:  # hard label
                    y_real = torch.ones(imgs.shape[0])
                    y_fake = torch.zeros(imgs.shape[0])

                # add noise labels
                y_noise_idx = torch.rand(imgs.shape[0]) < self.get_flip_y_freq()
                y_real.masked_fill_(y_noise_idx, 0)
                y_fake.masked_fill_(y_noise_idx, 1)

                if self.use_cuda:
                    imgs, lbls = imgs.cuda(), lbls.cuda()
                    t_imgs, t_lbls = t_imgs.cuda(), t_lbls.cuda()
                    y_real, y_fake = y_real.cuda(), y_fake.cuda()

                '''model forwarding & loss calculation'''
                # src
                src_cls_out, src_dom_out = self.model(imgs, self.get_alpha())
                src_cls_loss = self.criterion_cls(src_cls_out, lbls)
                src_dom_loss = self.criterion_bce(src_dom_out, y_fake)

                # tgt
                tgt_cls_out, tgt_dom_out = self.model(t_imgs, self.get_alpha())
                tgt_dom_loss = self.criterion_bce(tgt_dom_out, y_real)

                loss = src_cls_loss + src_dom_loss + tgt_dom_loss
                loss.backward()
                self.optim.step()

                with torch.no_grad():
                    _, preds = torch.max(tgt_cls_out, dim=1)
                    self.total_cnt += preds.cpu().numpy().size
                    self.correct_cnt += (preds == t_lbls).sum().item()

                self.writer.add_scalar('loss', loss.item(), self.iters)
                if current_iter % 50 == 0 or current_iter == len(self.train_loader):
                    print('Epoch [{}][{}/{}], Loss: {:.4f}'.format(epoch, current_iter, len(self.train_loader),
                                                                   loss.item()))

            train_acc = self.correct_cnt / self.total_cnt
            self.writer.add_scalar('acc/train_acc', train_acc, epoch)
            print('Epoch {}, Train Acc: {:.4f}'.format(epoch, train_acc))

            if epoch % self.val_epoch == 0:
                val_acc = self.evaluate()
                self.writer.add_scalar('acc/val_acc', val_acc, epoch)
                print('Epoch {}, Val Acc: {:.4f}'.format(epoch, val_acc))

                if val_acc > self.best_acc:
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_best.pth.tar'))
                    self.best_acc = val_acc

            if epoch % self.save_epoch == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_{}.pth.tar'.format(epoch)))
        print('\nBest Val Acc: {:.4f}\n'.format(self.best_acc))

    def evaluate(self):
        self.model.eval()
        total_cnt = 0
        correct_cnt = 0
        with torch.no_grad():  # do not need to calculate information for gradient during eval
            for idx, (imgs, gt, _) in enumerate(self.val_loader):
                if self.use_cuda:
                    imgs, gt = imgs.cuda(), gt.cuda()
                cls_out, _ = self.model(imgs, 0)
                _, pred = torch.max(cls_out, dim=1)

                total_cnt += pred.cpu().numpy().size
                correct_cnt += (pred == gt).sum().item()

        return correct_cnt / total_cnt
