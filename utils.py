import torch
import copy 


class AverageAcc(object):
    '''
    Computes top1 and top5 accuracies per class, and stores values (sum and count) in order to compute the average.
    '''

    def __init__(self, cdict):
        self.avg = cdict
        self.reset()


    def reset(self):
        self.sum = copy.deepcopy(self.avg)
        self.count = copy.deepcopy(self.avg)

    def accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # topk per class
        for c in self.avg:
            mt = (target == c).to(torch.long)
            mp = (pred == c).to(torch.long)
            mul = mp * mt

            correct = mul
            batch_size = sum(mt).item()
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0)
                self.sum[c][k] += correct_k.item()
                self.count[c][k] += sum(mt).item()


    def average(self):

        avg1_list = []
        avg5_list = []
        for k,v in self.avg.items():

            v[1] = round(self.sum[k][1]/self.count[k][1], 2)
            v[5] = round(self.sum[k][5]/self.count[k][5], 2)

            avg1_list.append(v[1])
            avg5_list.append(v[5])

        avg1 = sum(avg1_list)/len(avg1_list)
        avg5 = sum(avg5_list)/len(avg5_list)

        return self.avg, avg1, avg5
