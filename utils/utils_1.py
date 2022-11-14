import enum

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import os


class Custom_dataset(Dataset):
    def __init__(self, path1, path2, transform):
        self.real = [file.strip() for file in open(path1, 'r')]
        self.syn = [y.strip() for y in open(path2, 'r')]
        self.len = len(self.real)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        dict = {}
        y = self.syn[index]
        x = self.real[index]
        img1 = cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(y), cv2.COLOR_BGR2RGB)
        # image1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
        # image2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
        image1 = self.transform(img1)
        image2 = self.transform(img2)
        dict['A'] = image1
        dict['B'] = image2
        return dict


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    path1 = r'/home/moravapa/Documents/Thesis/synthetic/data/train1.txt'
    path2 = r'/home/moravapa/Documents/Thesis/synthetic/data/train2.txt'
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((512, 512)), transforms.ToTensor()])
    x = Custom_dataset(path1, path2, transform=transform)
    loader = DataLoader(x, batch_size=2, shuffle=False)
    print('loader', len(loader))
    for i, img in enumerate(loader):
        print('A', img['A'].shape)

        plt.imshow(img['A'][1].permute(1, 2, 0))
        plt.show()

        break
        print('B', img['B'].shape)
