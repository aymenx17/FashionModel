import os
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from model.net import FashionNet
from load_dataset import custom_dset, collate_fn, denorm
from focal_loss import *
from torch.autograd import Variable
import time
import cv2
import random
from tensorboardX import SummaryWriter
from utils import AverageMeter, accuracy

'''
We use the focal loss as cost function.

'''




writer = SummaryWriter(comment='_pretrain')

out_path = './data/show_validation'
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
                # save to disk first batch of results for every epoch
                for i,im in enumerate(imgs):
                    im = im.data.cpu().numpy()
                    im = denorm(im.transpose(1,2,0))[...,::-1]

                    _, inds = torch.max(preds, 1)
                    pred = label_map[inds[i].item()]
                    gt_label = label_map[labels[i].item()]

                    fname = 'Epoch:' + str(e + 1) +  'Gt:' + gt_label + 'Pred:' + pred + '.jpg'
                    cv2.imwrite(os.path.join(out_path, fname),  im)

    return val_loss

def train(epochs, net, train_loader, val_loader, optimizer,
          save_step):

    #crit = torch.nn.CrossEntropyLoss()
    crit = FocalLoss(gamma=2, alpha=0.25)
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

            # apply loss function
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
            state = {'net': net.state_dict(), 'label_map': train_loader.dataset.label_map }
            torch.save(state, './checkpoints/pnet_{}.pth'.format(e + 1))

def main():

    # Load dataset
    trainset = custom_dset('train_pretrain')
    valset = custom_dset('val_pretrain')

    print('Lenght of the training dataset: {}'.format(len(trainset)))
    print('Lenght of the validation dataset: {}'.format(len(valset)))
    train_loader = DataLoader(trainset, batch_size=28, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(valset, batch_size=28, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # FashionNet model
    num_classes = len(trainset.label_map)
    net = FashionNet(num_classes)
    net = net.cuda()
    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)


    train(epochs=11, net=net, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
               save_step=1)

    writer.close()



if __name__ == "__main__":
    main()
