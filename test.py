import torch
import torch.nn as nn
import os
import numpy as np
from model.net import FashionNet
from load_dataset import custom_dset, collate_fn, denorm
from torch.utils.data import DataLoader
import cv2
import argparse
from utils import AverageMeter, AverageAcc
from tqdm import tqdm
from focal_loss import *


def test(net, test_loader):


    losses = AverageMeter()

    class_dict = {v:{1: 0, 5:0} for _,v in test_loader.dataset.label_map.items()}
    acc = AverageAcc(class_dict)



    criterion = FocalLoss(gamma=2, alpha=0.25)

    bar = tqdm(total=len(test_loader))

    for it, (imgs, labels) in enumerate(test_loader):

        imgs = imgs.cuda()
        labels = labels.cuda()

        # run input through the network
        preds = net(imgs)
        loss = criterion(preds, labels)


        # measure accuracy and record loss
        acc.accuracy(preds.data, labels.data, topk=(1, 5))
        losses.update(loss.item(), imgs.size(0))

        bar.update()

    bar.close()

    topk_dict, top1, top5 = acc.average()
    return top1, top5, topk_dict



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

    top1, top5, topk_dict = test(net, test_loader)
    label_dict = { v:k for k,v in label_map.items()}

    print('Top1: {} and Top5: {}\n\n'.format(round(top1, 1), round(top5, 1)))

    # print directory of top1 and top5 per class
    for c,v in topk_dict.items():
        print('Class: {}     Top1: {}         Top5: {}'.format(label_dict[c], v[1], v[5]))


if __name__ == '__main__':
    main()
