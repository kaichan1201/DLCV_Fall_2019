from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='My parser for segmentation')

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
                        help="initial learning rate")

    # resume trained model
    parser.add_argument('--pretrained', type=str, default='',
                        help="path to the trained model")
    # others
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--random_seed', type=int, default=999)
    parser.add_argument('--use_model', type=str, default='BASE',
                        help='model to use')

    # for inference
    parser.add_argument('--infer_img', type=str, default='')

    args = parser.parse_args()

    return args
