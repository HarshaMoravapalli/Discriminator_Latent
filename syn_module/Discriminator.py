import torch.nn as nn
import torch
import functools
#from torchsummary import summary

#https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/551117b13ff017d852d067a7fa76e138d43dada5/models/networks.py#L4

class Discriminator(nn.Module):
    def __init__(self, args, num_filters_last=128, n_layers=3):
        super(Discriminator, self).__init__()
        kw =3

        layers = [nn.Conv2d(256, num_filters_last, kw, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last // num_filters_mult_last, num_filters_last // num_filters_mult, kw,
                          2 if i < n_layers else 1, 1, bias=False),
                nn.BatchNorm2d(num_filters_last // num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(nn.Conv2d(num_filters_last // num_filters_mult, 1, kw, 1, 1))
        #layers.append(nn.Conv2d(32, 1, 4, 1, 0))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)