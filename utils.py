import torch
import copy
import numpy as np


class AverageAcc(object):
    '''
    Stores variables like (sum and count), and computes top1 and top5 accuracies per class.


    {0: {1: 0, 5: 0},
     1: {1: 0, 5: 0},
     2: {1: 0, 5: 0},
     3: {1: 0, 5: 0},
     4: {1: 0, 5: 0},
     5: {1: 0, 5: 0},
     6: {1: 0, 5: 0},
     7: {1: 0, 5: 0},
     8: {1: 0, 5: 0},
     9: {1: 0, 5: 0},
     10: {1: 0, 5: 0},
     11: {1: 0, 5: 0},
     12: {1: 0, 5: 0},
     13: {1: 0, 5: 0},
     14: {1: 0, 5: 0},
     15: {1: 0, 5: 0},
     16: {1: 0, 5: 0},
     17: {1: 0, 5: 0},
     18: {1: 0, 5: 0}}



    '''

    def __init__(self, label_map):
        self.acc = {v:{1: 0, 5:0} for _,v in label_map.items()}
        self.reset()

    def __call__(self, *input, **kwargs):
        result = self.accuracy(*input, **kwargs)
        return result

    def reset(self):
        self.sum = copy.deepcopy(self.acc)
        self.count = copy.deepcopy(self.acc)

    def update(self, output, target, topk=(1,)):

        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        # topk per class
        for c in self.acc:
            # mask target and predictions and perform then intersection in between
            mt = (target == c).to(torch.long)
            mp = (pred == c).to(torch.long)
            correct = mp * mt

            # topk = (1, 5)
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0)
                self.sum[c][k] += correct_k.item()
                self.count[c][k] += sum(mt).item()


    def accuracy(self):

        avg1, avg5 = [], []
        for k,v in self.acc.items():
            # accuracy per class
            v[1] = round(self.sum[k][1]/self.count[k][1], 2)
            v[5] = round(self.sum[k][5]/self.count[k][5], 2)

            avg1.append(v[1])
            avg5.append(v[5])

        return self.acc, np.mean(avg1), np.mean(avg5)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
