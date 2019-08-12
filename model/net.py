
import torch
import torch.nn as nn
from model.resnet import resnet50
import capsules as caps



class FashionNet(nn.Module):

    def __init__(self, num_classes=19, channels=512, primary_dim=16, out_dim=32,  num_routing=3):
        super(FashionNet, self).__init__()

        self.resnet = resnet50(True)
        self.primary = caps.PrimaryCapsules(channels, channels, primary_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.digits = caps.RoutingCapsules(primary_dim, 2560, num_classes, out_dim, num_routing, device = self.device )

    def forward(self, x):

        x = self.resnet(x)
        out = self.primary(x)
        out = self.digits(out)
        preds = torch.norm(out, dim=-1)

        return preds
