import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horizontal_flip
from torch.utils.data import Dataset
from torchvision import transforms

# description :  # 동일 사이즈로 패딩해준다. 3x5 -> 5x5되고 사이즈 동일하게 패딩
def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # 이미지의 height - width 를 뺀다.
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2 , dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, int(pad1), int(pad2)) if h <= w else (int(pad1), int(pad2), 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad

# description : (1,5,3) 와 같은 이미지를 넣으면 (1, size, size)로 resize 해준다.
# mode는 nearest로...
def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

# description : random.sample
# list(range(0,9,2)), 2는 간격
# [0, 2, 4, 6, 8]
# 그래서 첫번째 인자로 list를 넣어주고 다음 인자로 몇개를 뽑을 것인가?
def random_resize(images, min_size=288, max_size=488):
    new_size = random.sample(list(range(min_size, max_size+1, 32)), 1)[0]
    images= F.interpolate(images, size=new_size, mode="nearest")
    return images

# import os
# >>> os.getcwd() => 현재 디렉토리 위치
# glob.glob("folder_path/*.*")이렇게 하면 해당 폴더안의 파일들을 다 찾을 수 있다.
class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path)) # 한방에 그리고 sorted를 씀으로써 정렬함@
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)] # 저렇게 %를 붙임으로써 max를 한정
        # Extract image as Pytorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to Square Resolution
        img, _ = pad_to_square(img, 0)
        #Resize
        img = resize(img, self.img_size)
        return img_path, img

    def __len__(self):
        return len(self.files)

class ListDataset(Dataset):
    def __init__(self, list_path, img_size=446, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file: # file List에 대한 Path를 받았다. 
            self.img_files = file.readlines() # Line들을 List 형태로 보관

        self.label_files = [
            path.replace("images",
                         "labels").replace(".png",
                                            ".txt").replace(".png",
                                                             ".txt").replace(".jpg",
                                                                              ".txt")
            for path in self.img_files
        ]

        self.img_size = img_size                   # image의 사이즈
        self.max_objects = 100 
        self.augment = augment                     # augment를 할 까?
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels 
        self.min_size =  self.img_size - 3 * 32
        self.max_size =  self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):
        #-------#
        # Image #
        #-------#
        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as Pytorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle Images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w= img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        #Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w =img.shape

        #-------#
        # Lable #
        #-------#
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1,5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:,3]/2)
            y1 = h_factor * (boxes[:, 2] - boxes[:,4]/2)
            x2 = w_factor * (boxes[:, 2] - boxes[:,3]/2)
            y2 = h_factor * (boxes[:, 2] - boxes[:,4]/2)
            # Adjust for added Padding
            x1 +=pad[0]
            y1 +=pad[2]
            x2 +=pad[1]
            y2 +=pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) /2) / padded_w
            boxes[:, 2] = ((y1 + y2) /2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes) ,6))
            targets[:, 1:] = boxes
        print("-----------------------------")
        # Apply augmentations
        if self.augment :
            if np.random.random() < 0.5:
                img, targets = horizontal_flip(img, targets)

        return img_path, img, targets


if __name__ == "__main__" :
    a= torch.tensor(np.ones((5,3))).unsqueeze(0)
    print(a.shape)
    print(random_resize(a).shape)

'''
# pad_to_square -> Testing 1
if __name__ == "__main__" :
    a= torch.tensor(np.ones((5,3)) )
    a=a.unsqueeze(0)
    print(a.shape)
    T = F.pad(a,(0,0,2,3)) # 위, 아래 패딩
    print(T, T.shape)
    R = F.pad(a,(2,3,0,0)) # 왼, 오른쪽 패딩
    print(R, R.shape)
    print(">>>>",pad_to_square(a,6))
   
'''

