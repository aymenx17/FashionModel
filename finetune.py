import os
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from model.net import FashionNet
from load_dataset import custom_dset, collate_fn, denorm
from torch.autograd import Variable
import time
import cv2
import random
from tensorboardX import SummaryWriter
from utils import AverageMeter, accuracy

'''
The objective of this module is to finetune the previously trained network, on a new set of classes, on which
there hasn't been trained before. We apply the concept of transfer leaning, so it is not required to train
the whole network from scratch for the new task, instead we transfer the knowledge (weight parameters) and only
train for instance the last layer of the network. Thefore the principle behind is to share parameters
for different tasks; in this context tasks means different set of classes.

'''

writer = SummaryWriter(comment='finetune')

out_path = './data/finetune_show'
if not os.path.isdir(out_path):
    os.mkdir(out_path)


def eval_val(e, val_loader, crit, net):

    val_loss = AverageMeter()
    label_map = val_loader.dataset.label_map
    label_map = { v:k for k,v in label_map.items()}

    with torch.no_grad():
        for it, (imgs, lables) in enumerate(val_loader):


            imgs = imgs.cuda()
            labels = lables.cuda()
            # run input through the network
            preds = net(imgs)

            loss = crit(preds, labels)
            val_loss.update(loss.item(), imgs.size()[0])

            if it == 0:
                # save a batch of results to disk
                for i,im in enumerate(imgs):
                    im = im.data.cpu().numpy()
                    im = denorm(im.transpose(1,2,0))[...,::-1]


                    _, inds = torch.max(preds, 1)
                    pred = label_map[inds[i].item()]
                    gt_label = label_map[labels[i].item()]

                    fname = 'Epoch:' + str(e + 1) +  'Gt:' + gt_label + 'Pred:' + pred + '.jpg'
                    cv2.imwrite(os.path.join(out_path, fname),  im)

    return val_loss


def finetune(epochs, net, train_loader, val_loader, optimizer, save_step):

    crit = torch.nn.CrossEntropyLoss()
    train_loss = AverageMeter()

    for e in range(epochs):
        print('*'* 100)
        print('Epoch {} / {}'.format(e + 1, epochs))
        net.train()

        # training stage
        for it, (img, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            img = Variable(img.cuda())
            labels = Variable(labels.cuda())


            # run input through the network
            preds = net(img)

            # CrossEntropyLoss
            loss = crit(preds, labels)

            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), img.size()[0])

            if (it + 1) % 30 == 0:
                net.eval()
                val_loss  = eval_val(e, val_loader, crit, net)
                print('Epoch: {} Training loss: {} Validation loss: {} Step (it/tot): {}/{}'.format((e + 1), round(train_loss.avg, 3), round(val_loss.avg, 3), it, len(train_loader)))
                tot_iter = (e*len(train_loader) + it)
                writer.add_scalars('losses',{ 'train_loss': train_loss.avg, 'validation_loss': val_loss.avg}, tot_iter)
                net.train()


        if (e + 1) % save_step == 0:
            if not os.path.exists('./checkpoints'):
                os.mkdir('./checkpoints')
            torch.save(net.state_dict(), './checkpoints/finetuned_net_{}.pth'.format(e + 1))


def main():

    # Collect arguments (if any)
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default=None, help='path to the model')
    args = parser.parse_args()

    # get the current working directory
    root = os.getcwd()

    train_finetune = custom_dset('train_finetune')
    trainval_finetune = custom_dset('val_finetune')

    print('Train Dataset for finetuning of size: {}'.format(len(train_finetune)))
    print('Validation Dataset for finetuning of size: {}'.format(len(trainval_finetune)))
    train_loader = DataLoader(train_finetune, batch_size=28, shuffle=True, collate_fn=collate_fn, num_workers=4)
    trainval_loader = DataLoader(trainval_finetune, batch_size=28, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # number of classes we are going to finetune over
    num_classes = len(train_finetune.label_map)

    # load model
    model_path = os.path.join(root, 'checkpoints', args.model)

    model = torch.load(model_path)
    label_map = model['label_map']
    net = FashionNet(len(label_map))
    net.load_state_dict(model['net'])
    # replace the classifier
    net.fc = torch.nn.Linear(2048, num_classes)
    net = net.cuda()


    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    finetune(epochs=11, net=net, train_loader=train_loader, val_loader=trainval_loader, optimizer=optimizer,
               save_step=1)

    writer.close()


if __name__ == '__main__':
    main()
