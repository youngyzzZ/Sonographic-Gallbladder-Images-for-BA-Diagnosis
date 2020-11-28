# @ FileName: tool.py
# @ Author: Alexis
# @ Time: 20-11-28 下午9:17
import argparse
import os
from os.path import exists
from src import config
import numpy as np
import matplotlib.pyplot as plt
import itertools


def gen_parser():
    parser = argparse.ArgumentParser(prog=config.PROGRAM,
                                     description=config.DESCRIPTION)

    parser.add_argument('cmd', choices=config.cmd_list, help='what to do')
    parser.add_argument('--desc', type=str, default='', help='description')
    parser.add_argument('--action', type=str, default='base', help='action')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='ImageNet100',
                        help='dataset')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--model', type=str, default='res18', help='model')

    parser.add_argument('--train_csv', type=str, default='label_except_empty_all.csv', help='train label file')
    parser.add_argument('--test_csv', type=str, default='label.csv', help='test label file')

    # source setting
    parser.add_argument('--cuda', type=str, default='0', help='gpu(s)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num workers for dataloader')

    # basic setting
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_epoch', type=int, default=35,
                        help='learning rate decay epoch')
    parser.add_argument('--epoch', type=int, default=200, help='epoch')
    parser.add_argument('--pre_train', action='store_true', default=False,
                        help='pre train')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch for all')
    parser.add_argument('--weight', '-w', nargs='+', type=float, default=[5.0, 1.0])
    parser.add_argument('--save_cm', action='store_true', default=False,
                        help='save confusion matrix')

    # result
    parser.add_argument('--no_eval', action='store_true', default=False,
                        help='no need to eval')
    parser.add_argument('--personal_eval', action='store_true', default=False,
                        help='eval by personal')

    return parser.parse_args()


def check_mkdir(dir_name):
    if not exists(dir_name):
        os.makedirs(dir_name)


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure
