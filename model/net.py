
import torch
import torch.nn as nn
from model.resnet import resnet50




class FashionNet(nn.Module):

    def __init__(self, num_classes=20):
        super(FashionNet, self).__init__()

        self.resnet = resnet50(True)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):

        x = self.resnet(x)
        x = self.fc(x)

        return x
