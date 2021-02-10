import argparse
import os
import sys

sys.path.append(os.getcwd())

import torch
from tqdm import tqdm

from src.util.data.data_loader import load_subset
from src.util.dotdict import Dotdict
from toy.demo import setup, im_test
from src.util.validate import calc_scores


def get_opt():
    opt = Dotdict()

    opt.model = 'all'
    opt.is_train = True
    opt.pretrained = True
    opt.checkpoints_dir = './out/checkpoints/faces'
    opt.continue_train = True
    opt.save_name = 'latest'
    opt.name = 'knn'
    opt.dataset_path = './datasets/celeb-df-v2/images'
    # opt.dataset_path = './datasets/forensic/images'
    opt.multiclass = False
    opt.resize_interpolation = 'bilinear'
    opt.load_size = -1
    opt.train_split = 'train'
    opt.train_size = 2500
    opt.val_split = 'val'
    opt.val_size = 100
    opt.test_split = 'test'

    return opt


def test(args):
    net = setup(args)
    model_path = os.path.join(args.save_dir, args.ckpt_name)
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_path, start_epoch))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(model_path))
    opt = get_opt()
    test_img, test_label = load_subset(opt, opt.test_split, opt.load_size)
    y_pred = []
    for img in tqdm(test_img):
        pred = im_test(net, img, args)[0]
        y_pred.append(1 - pred)  # returns real prob, I use fake prob
    acc, ap, auc = calc_scores(test_label, y_pred)[:3]
    print("Test: acc: {}; ap: {}; auc: {}".format(acc, ap, auc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='')
    parser.add_argument('--arch', type=str, default='sppnet',
                        help='VGG, ResNet, SqueezeNet, DenseNet, InceptionNet')
    parser.add_argument('--layers', type=int, default='50')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--save_dir', type=str, default='./src/baselines/dsp_fwa/ckpt/')
    parser.add_argument('--ckpt_name', type=str, default='SPP-res-50.pth')
    test(parser.parse_args())
