from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='My parser for GAN')

    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='data',
                        help="root path to data directory")
    parser.add_argument('--workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")

    # training parameters
    parser.add_argument('--epochs', '-e', default=100, type=int,
                        help="num of validation iterations")
    parser.add_argument('--val_epoch', default=10, type=int,
                        help="num of validation iterations")
    parser.add_argument('--save_epoch', default=10, type=int,
                        help="num of save iterations")
    parser.add_argument('--batch_size', '-b', default=32, type=int,
                        help="batch size")
    parser.add_argument('--lr', '-lr', default=0.0002, type=float,
                        help="initial learning rate")
    parser.add_argument('--weight_decay', default=0.0005, type=float,
                        help="rate of weight decay")
    parser.add_argument('--soft_label', action='store_true',
                        help="use soft label")
    parser.add_argument('--flip_y_freq', default=0, type=float,
                        help="freq of noisy y")
    parser.add_argument('--n_classes', default=10, type=int,
                        help="# of classes")

    # resume trained model
    parser.add_argument('--pretrained_G', type=str, default='',
                        help="path to the trained G")
    parser.add_argument('--pretrained_D', type=str, default='',
                        help="path to the trained D")
    parser.add_argument('--pretrained', type=str, default='',
                        help="path to the trained model")
    # others
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--random_seed', type=int, default=999)

    # model
    parser.add_argument('--nz', default=100, type=int,
                        help="size of the latent z vector")
    parser.add_argument('--ngf', default=64, type=int,
                        help="size of the G feature vector")
    parser.add_argument('--ndf', default=64, type=int,
                        help="size of the D feature vector")

    # for DANN
    parser.add_argument('--src', type=str, default='',
                        help="source data")
    parser.add_argument('--tgt', type=str, default='',
                        help="target data")
    parser.add_argument('--no_transfer', action='store_true',
                        help="whether transfer or not")

    # for DTN
    parser.add_argument('--pretrained_cls', type=str, default='',
                        help="load pretrained classifier model for evaluation")
    parser.add_argument('--noisy_input', type=float, default=0,
                        help="add noise to real inputs (number: std of noise)")

    # for ADDA
    parser.add_argument('--pretrained_src', type=str, default='',
                        help="load pretrained source model")

    # for inference
    parser.add_argument('--infer_img', type=str, default='')
    parser.add_argument('--infer_GAN_mode', type=str, default='DC', choices=['DC', 'AC'])
    parser.add_argument('--infer_data_dir', type=str, default='')
    parser.add_argument('--csv_path', type=str, default='')

    # for tSNE
    parser.add_argument('--tSNE_model', type=str, default='DANN', choices=['DANN', 'ADDA'])

    args = parser.parse_args()

    return args
