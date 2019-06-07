import torch
import torch.nn.functional as F
import numpy as np


def horizontal_flip(images, targets):
    print(images.shape)
    images= torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


# TDD : Test Phase  나주엥 Test #
if __name__ == "__main__" :
    img=torch.tensor(np.arange(300));
    tar=torch.tensor(np.arange(1));
    img=img.view((10,10,3))

    img=img.unsqueeze(0)
    tar=tar.unsqueeze(0)
    print(img.shape, tar.shape)
    horizontal_flip(img,tar)
