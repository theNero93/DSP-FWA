import argparse
import os
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from src.data.basic_dataset import BasicDataset
from src.model.transforms.transform_builder import create_basic_transforms
from src.util.validate import calc_scores, save_pred_to_csv
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
    dataset_dir = join(args.root_dir, args.dataset)
    test_data = BasicDataset(root_dir=dataset_dir,
                             processed_dir=args.processed_dir,
                             crops_dir=args.crops_dir,
                             split_csv=args.split_csv,
                             seed=args.seed,
                             normalize=None,
                             transforms=create_basic_transforms(args.input_size),
                             mode='test')
    test_loader = DataLoader(test_data, batch_size=1)
    y_pred, y_real = [], []
    i = 0
    for img, label in tqdm(test_loader):
        pred = 1 - im_test(net, np.asarray(transforms.ToPILImage()(img.squeeze(0))), args)[
            0]  # returns real prob, I use fake prob
        if pred > 1.0:
            continue
        y_pred.append(pred)
        y_real.append(label.cpu().item())
    acc, ap, auc = calc_scores(y_real, y_pred)[:3]
    print("Test: acc: {}; AUC: {}; loss: {}".format(acc, ap, auc))
    if args.save_pred:
        save_pred_to_csv(y_real, y_pred, args.name, args.dataset)


def parse_args():
    parser = argparse.ArgumentParser(description="Parameters for Training")
    args = parser.add_argument
    # Dataset Options
    args("--root_dir", default='/bigssd/datasets', help="root directory")
    args('--dataset', default='dfdc')
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
    args('--name', default='bl_dsp_fwa_resnet_50')
    args('--save_pred', default=True)
    return parser.parse_args()


if __name__ == '__main__':
    test(parse_args())
