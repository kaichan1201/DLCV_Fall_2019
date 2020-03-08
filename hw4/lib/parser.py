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
    parser.add_argument('--val_epoch', default=1, type=int,
                        help="num of validation iterations")
    parser.add_argument('--save_epoch', default=10, type=int,
                        help="num of save iterations")
    parser.add_argument('--batch_size', '-b', default=32, type=int,
                        help="batch size")
    parser.add_argument('--lr', '-lr', default=0.0002, type=float,
                        help="initial learning rate")
    parser.add_argument('--weight_decay', default=0.0005, type=float,
                        help="rate of weight decay")

    # RNN
    parser.add_argument('--rnn_batch_num', default=4, type=int,
                        help="# of videos to train rnn in 1 iteration")

    # resume trained model
    parser.add_argument('--pretrained', type=str, default='',
                        help="path to the trained model")

    # data
    parser.add_argument('--downsample', type=int, default=12,
                        help='factor to downsample the video (fps = 24 / d)')
    parser.add_argument('--rescale', type=float, default=1,
                        help='factor to rescale the frames')
    parser.add_argument('--frame_num', type=int, default=4,
                        help='# of frames to use for input')

    # others
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--random_seed', type=int, default=999)

    # model
    parser.add_argument('--n_classes', default=10, type=int,
                        help="# of classes")

    # infer
    parser.add_argument('--infer_data_dir', type=str, default='')
    parser.add_argument('--csv_path', type=str, default='')

    # for tSNE
    parser.add_argument('--tSNE_model', type=str, default='DANN', choices=['DANN', 'ADDA'])

    args = parser.parse_args()

    return args
