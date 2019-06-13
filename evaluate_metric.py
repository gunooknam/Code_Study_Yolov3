from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get DataLoader
    dataset = ListDataset(path, img_size=img_size, augment=False)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )
...
