import torch
import torch.nn as nn
import os
import numpy as np
from model.net import FashionNet
from load_dataset import custom_dset, collate_fn, denorm
from torch.utils.data import DataLoader
import cv2
import argparse
from utils import AverageMeter, accuracy
from tqdm import tqdm
from focal_loss import *


def test(net, test_loader):


    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    criterion = FocalLoss(gamma=2, alpha=0.25)

    bar = tqdm(total=len(test_loader))

    for it, (imgs, labels) in enumerate(test_loader):

        imgs = imgs.cuda()
        labels = labels.cuda()

        # run input through the network
        preds = net(imgs)
        loss = criterion(preds, labels)


        # measure accuracy and record loss
        prec1, prec5 = accuracy(preds.data, labels.data, topk=(1, 5))
        losses.update(loss.item(), imgs.size(0))
        top1.update(prec1.item(), imgs.size(0))
        top5.update(prec5.item(), imgs.size(0))

        bar.update()

    bar.close()
    return top1.avg, top5.avg



def main():

    # Collect arguments (if any)
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default=None, help='path to the model')
    args = parser.parse_args()

    # get the current working directory
    root = os.getcwd()

    # load model
    model_path = os.path.join(root, 'checkpoints', args.model)
    model = torch.load(model_path)
    label_map = model['label_map']
    net = FashionNet(len(label_map))
    net.load_state_dict(model['net'])

    net = net.cuda()
    net.eval()


    test_pretrain = custom_dset('test_pretrain', label_map)
    print('Pretrain Test Dataset of size: {}'.format(len(test_pretrain)))
    test_loader = DataLoader(test_pretrain, batch_size=28, shuffle=False, collate_fn=collate_fn, num_workers=4)

    top1, top5 = test(net, test_loader)

    print('Top1: {} and Top5: {}'.format(round(top1, 1), round(top5, 1)))



if __name__ == '__main__':
    main()
