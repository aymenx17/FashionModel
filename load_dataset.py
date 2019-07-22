import json
import os
from torch.utils import data
import numpy as np
import torch
import cv2
import collections

'''
Notes on the small fashion dataset:

The class 'Perfume and Body Mist' is present only in the test set since all examples from this class correspond an odd year (2017).

There is a similar problem during finetuning, where training set has 88 classes and test set has 99 classes.
Therefore the classes not in common between the two sets are unused.



'''



root_img = os.path.join(os.getcwd(),'data', 'fashion-product-images-small', 'images')
json_path = './data/fashion-product-images-small/styles.json'

with open(json_path) as f:
    jfile = json.load(f)


def denorm(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return np.uint8((tensor*std + mean)*255)

def load_ids(dset=None, test_labels=None):

    ''' Load a list of data ids for a specific type of dataset.
        For the task of transfer learning, the fashion dataset is split into six group of data examples without overlapping.
        Types:
              Training:

                - train_pretrain
                - val_pretrain
                - train_finetune
                - val_finetune

              Test:

                - test_pretrain
                - test_finetune
    '''
    top20 = ['Tshirts', 'Shirts', 'Casual Shoes', 'Watches', 'Sports Shoes', 'Kurtas', 'Tops', 'Handbags', 'Heels', 'Sunglasses',
            'Wallets', 'Flip Flops', 'Sandals', 'Briefs', 'Belts', 'Backpacks', 'Socks', 'Formal Shoes', 'Perfume and Body Mist', 'Jeans']

    lids = []
    categs = []
    if test_labels == None:
        test_labels = []

    with open(json_path) as f:
        data = json.load(f)
        for k, d in data.items():
            year = d['year']
            categ = d['categ']
            fname = k + '.jpg'
            if year.isdigit():
                p_img = os.path.join(root_img, fname)
                year = int(year)

                # check if the image is present
                if os.path.isfile(p_img):
                    if year % 2 == 0 and (categ in top20) and (dset == 'train_pretrain' or dset == 'val_pretrain'):
                        lids.append(k)
                        categs.append(categ)
                    elif year % 2 == 0 and (categ not in top20) and (dset == 'train_finetune' or dset == 'val_finetune'):
                        lids.append(k)
                        categs.append(categ)
                    elif categ in test_labels and year % 2 != 0 and (categ in top20) and dset == 'test_pretrain':
                        lids.append(k)
                        categs.append(categ)
                    elif categ in test_labels and year % 2 != 0 and (categ not in top20) and dset == 'test_finetune':
                        lids.append(k)
                        categs.append(categ)


    # we can give an order to the classes
    freq = collections.Counter(categs)
    mc = freq.most_common(len(freq))
    categs = [t[0] for t in mc]
    label_map = {categ:n for n, categ in enumerate(categs)}

    val_size = round(len(lids)*0.15)
    if dset == 'val_pretrain' or dset == 'val_finetune':
        lids = lids[:val_size]
    elif dset == 'train_pretrain' or dset == 'train_finetune':
        lids = lids[val_size:]

    # Since test and train sets may have some classes not in common
    if dset == 'test_pretrain' or dset == 'test_finetune':
        label_map = test_labels



    return lids, label_map




def get_data(image_list, label_map, index):

    try:

        # read one image
        img_id = image_list[index]
        img_p = os.path.join(root_img, img_id + '.jpg')
        img = cv2.imread(img_p)[...,::-1]
        # pytorch pretrained models work with Resolution of at least 224x224.
        # However I made a change on resnet file --> nn.AvgPool2d(2). Now is possible to train on lower Resolution.
        img = cv2.resize(img, (60, 80))

        # read the correspondent annotation
        categ = jfile[img_id]['categ']
        label = label_map[categ]


    except Exception  as e:
        img, label = None, None
        print('Exception in get_data()')

    # scale to [0 1] and normalize the input image.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img / 255 - mean) / std
    img = img.astype(np.float32)

    return img, label

class custom_dset(data.Dataset):
    def __init__(self, dataset, test_labels=None):
        self.image_list, self.label_map  = load_ids(dataset, test_labels)
    def __getitem__(self, index):
        img, anns = get_data(self.image_list, self.label_map, index)
        return img, anns

    def __len__(self):
        return len(self.image_list)


def collate_fn(batch):
    img, anns = zip(*batch)
    images = []

    for i in range(len(img)):
        if img[i] is not None:
            a = torch.from_numpy(img[i])
            a = a.permute(2, 0, 1)
            images.append(a)


    images = torch.stack(images, 0)
    labels = torch.LongTensor(anns)

    return images, labels
