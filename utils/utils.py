from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patchs

def to_cpu(tensor):
    return tensor.detach().cpu()

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    # -1까지 하는 이유 마지막에 공백이 있다.
    print("Load Class Nums : ",len(names))
    return names

# weight initialize....!
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    # Rescales bounding Boxes to the original shape
    orig_h, orig_w = original_shape # 원래의 shape을 넣는다.
    # The Amount of Padding That was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_w, 0) * (current_dim / max(original_shape))
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale Bounding Boxes to Dimension of Original Image
    boxes[:, 0] = ((boxes[:, 0] - pad_x //2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y //2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x //2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y //2) / unpad_w) * orig_h
    return boxes


# xywh
# x,y는 중심의 좌표이다. w,h는 width와 height이다. 
# xyxy => # left top, right bottom

def xywh2xyxy(x):
    y = x.new(x.shape) # torch의 tensor에는 new라는 함수가 있음 그냥 x.shape정도로 tensor 만드는 듯
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def ap_per_class(tp, conf, pred_cls, target_cls): # Compute the average precision, given the recall and precision curves.
    # => Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Args
    #   tp: True positives (list)
    #   conf : Objectness value from 0-1 (list)
    #   pred_cls : Predicted object classes (list)
    #   target_cls : True object classes (list)
    # Returns
    #   The average precision as computed in py-faster-rcnn
    # argsort는 index 순으로 sort 하는 것 L=[5,2,3,5,6] # >>> np.argsort(L) # array([1, 2, 0, 3, 4], dtype=int64)
    # np.argsort(L) 안에 L이라는 것에다가 -를 붙이면 역순 출력
    i = np.argsort(-conf)
    tp ,conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls) # True Object의 클래스이다.

    # Create Precision-Recall curve and compute Ap for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum() # Number of ground truth objects
        n_p = i.sum() # Number of predicted objects


def compute_ap(recall, precision):
    '''
    # Args
        recall : The recall curve (list)   # recall과 precision의 index의 구간 개수는 같을 것이다. 
        precision : The precision curve (list)
    # Returns
        The average precision as computed in py-faster-rcnn.
    '''
    # correct AP calculation
    # first append sentinel values at the end
    # 가로축은 Recall 이고
    # 세로축은 Precision 이길 원한다.
    mrec = np.concatenate(([0.0], recall, [1.0])) # recall 에 관한 curve List이다.
    mpre = np.concatenate(([0,0], recall, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size -1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # 큰수 부터 역순
    # ex). mpre.size 10이면 10, 9 ,8, 7, 6, 5, 4, 3, 2,...

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0] # 다른 부분의 index를 리턴

    # and sum(\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i+1] ) # i는 인덱스이기 때문에 
    # 이부분은 적분 같은 느낌이다.
    # mrec[i+1] - mrec[i]는 구간 사이값들의 리스트이다.
    # 이것을 마찬가지로 구간 높이값인 mpre[i+1]들과 곱하여 이것의 합을 구하는 것 => AUC임
    return ap

def get_batch_statistics(outputs, targets, iou_threshold):
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[: , :4]
        



# Testing
if __name__ == "__main__" :
    classes_defs=load_classes("../data/coco.names")
    print(classes_defs)

