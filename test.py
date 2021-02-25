import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.basic_dataset import BasicDataset
from src.data.transforms import create_basic_transforms
from src.util.validate import calc_scores
from toy.demo import setup, im_test


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
    test_data = BasicDataset(root_dir=args.root_dir,
                             processed_dir=args.processed_dir,
                             crops_dir=args.crops_dir,
                             split_csv=args.split_csv,
                             seed=args.seed,
                             normalize=None,
                             transforms=create_basic_transforms(args.size),
                             mode='test')
    test_loader = DataLoader(test_data, batch_size=1)
    y_pred, y_real = [], []
    for img, label in tqdm(test_loader):
        pred = im_test(net, img.numpy(), args)[0]
        y_pred.append(1 - pred)  # returns real prob, I use fake prob
        y_real.append(y_real)
    acc, ap, auc = calc_scores(y_real, y_pred)[:3]
    print("Test: acc: {}; ap: {}; auc: {}".format(acc, ap, auc))


def parse_args():
    parser = argparse.ArgumentParser(description="Parameters for Training")
    args = parser.add_argument
    # Dataset Options
    args("--root_dir", default='datasets/dfdc', help="root directory")
    args('--processed_dir', default='processed', help='directory where the processed files are stored')
    args('--crops_dir', default='crops', help='directory of the crops')
    args('--split_csv', default='folds.csv', help='Split CSV Filename')
    args('--seed', default=111, help='Random Seed')

    args('--input', type=str, default='')
    args('--arch', type=str, default='sppnet', help='VGG, ResNet, SqueezeNet, DenseNet, InceptionNet')
    args('--layers', type=int, default='50')
    args('--input_size', type=int, default=224)
    args('--save_dir', type=str, default='./src/baselines/dsp_fwa/ckpt/')
    args('--ckpt_name', type=str, default='SPP-res-50.pth')
    return parser.parse_args()


if __name__ == '__main__':
    test(parse_args())
