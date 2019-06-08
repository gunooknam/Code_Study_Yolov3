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
    """ Loads class labels at 'path'  """
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
def xywh2xyxy(x): # shape 상관없이 가장 끝쪽의 format을 변경
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
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = [] 
    for sample_i in range(len(outputs)): # output 은 배열형태

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes  = output[:, :4]
        pred_scores = output[:,  4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i] [:,1 :]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations) :
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                if len(detected_boxes) == len(annotations) :
                    break

                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]

            batch_metrics.append([true_positives, pred_scores, pred_labels])
        return batch_metrics

def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t() # torch.t()는 그냥 transpose임
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h2 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    => torch Tensor 타입이 안으로 들어온다. 단 type은 forch.float64 or float32
    Args : box1는 두가지 타입이 있다.   x1, y1, x2, y2
    Case1 : box1[Num of Batch의 개수,   0~3 ]이라면? => 기존 상태를 유지
                                        xc, yc, w, h
    Case2 : box1[Num of Batch의 개수,   0~3 ]이라면? => x1y1x2y2로 바꾼다.
    그래서 IOU를 구할 때는 left Top, right bottom의 두 좌표 형태로 바꿔야 한다.
    """
    if not x1y1x2y2: # 만약에 => xc,yc,w,h (xc,yc은 센터점), 이걸 left_top, right_bottom 으로 바꿈
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else : # 이미 left_top, right_bottom 으로 되어 있다면
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    print(b1_x1, b1_y1, b1_x2, b1_y2)
    print(b2_x1, b2_y1, b2_x2, b2_y2)

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp( inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1) # batch 통으로 한번에 다 구함
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1) # batch 통으로 한번에 다 구함
    print(b1_area, b2_area, inter_area)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou # batch 만큼의 iou가 담긴 List


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    # 이건 논 맥시멈 서프레션이란 기법인데 어떠한 threshold 값을 정하고
    이러한 threshold 보다 confidence score가 낮으면 답이 될 후보군에서 지워버리는 것이다.
    Returns dectections with shape:
        (x1, y1, x2, y2, obejct_conf, class_score, class_pred)
    """
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    # prediction도 batch단위로 있을 것이다.
    prediction[..., :4] = xywh2xyxy(prediction[..., :4]) # 코코면 이거 shape이 (1,개많음,85) 이정도 된다.
    # 이부분에선 iou 구하기 위한 top left, right bottom 형식으로 prediction이라는 tensor로 바꾸는 부분이다.

    output = [None for _ in range(len(prediction))] # => [None, None, None, None, None, None, None.... ]
    for image_i, image_pred in enumerate(prediction): # image index, image_pred 값을 뽑아낸당
        image_pred = image_pred[ image_pred[:,4] >= conf_thres ] # 요기서 개쪼금 추려진다. [10647, 85]-> [5,85] 정도?
        # image_pred[:,4] 라는 confidence 값이 conf_thres보다 큰 경우의 index를 추려낸다.
        #  -> 그 값을 가진 image_pred가 다시 추려내진다.
        # image_pred는 threshold에 의해서 걸러진 갯수로 뽑아진다.!!

        if not image_pred.size(0): # 그런데 만약에 size의 첫번째 rank가 0이라믄...
            continue

        # Object Confidence times class confidence
        # Object Confidence와 Class Confidence와는 다른 얘기다.
        # Object Confidence는 물체가 여기 Grid에 있을 까?
        # Class  Confidence는 어떤 클래스에 대한 확률이다.
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # 인덱스마다 가장 큰 class confidence를 뽑고 그것과 같은 index의 object confidence와 곱한다.
        # max(1) 이렇게 하면 2개 이상의 tensor들이 나옴 그중에 0번째(값들)를 고르는 것이다., 1번쨰(index)들!!

        #Sort By it             # 이 버전에서는 pytorch에서 argsort가 안되나부다...
        image_pred = image_pred[(-score).numpy().argsort()] # threshold로 인해서 5개가 추려졌다면 이걸 정렬해서 배치한다.
        # 그니까 index로 배열이 들어감 그러면 그 배열에 맞는 것만 추려짐
        # score가 큰 순대로 인덱스를 뽑아냄-> argsort() -> 값순으로 정렬조건이지만 index가 정렬된다.
        # 그 index에 맞는 것을 다시 image_pred에 넣으므로 score 내림 차순으로 image_pred가 재 배열된다.


                                            # 5~85개 까지의 class confidence 중에서 max를 뽑아낸다.
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)  # max를 찾고 rank 하나가 감소된다. But dimension을 유지하는 Trick을 썻다.
        # threshold로 인해서 5개가 추려졌다면 5개의 confidence에 대한 값들 max 추린것
        # threshold로 인해서 5개가 추려졌다면 몇번째 predict된 클래스인지의 class index

                            # Cat을 하는 것이다. xywh,class_confidence, class_label 이렇게 추려진다.
        detections = torch.cat((image_pred[:, :5],
                                class_confs.float(),  # class_index에 해당하는 confidence가 뭐니?
                                class_preds.float()), # 몇번의 class index니?
                                1)
                                # 마지막 인자는 dimension인데 어디를 합쳐줄지 정한다.
                                # 1을 한 이유는 => 앞의 batch는 남기는 것을 원한 것이다.

        # 그래서 결과는 threshold로 5개가 추려졌다면 (5,7) 의 tensor가 detection으로 뽑아내지는 것이다.

        ######## Perform non-maximum suppression #########
        keep_boxes = []
        while detections.size(0): # 첫번째 score가 높은 것과 [1,4] -> 이것과,   이전에 추려낸 [5,4]를 다 비교한다. 그래서 [5] 라는 iou가 나온다. => broadcast가 되는가 싶다.
            large_overlap=bbox_iou( detections[0, :4].unsqueeze(0), detections[:, :4])  > nms_thres # Non Maximum Suppression Parameter
            label_match = detections[0, -1] == detections[:, -1] # 그래서 score가 높은 것과 나머지 것들과 라벨이 같은 것을 추려낸다.
            # 예를 들어 detections[0, -1] 은 tensor(0.)
            #    그리고 detections[:, -1] 은 tensor([ 0.,  0., 17., 16., 17.]) 라고 하자 그러면 매치 되는 것은 [1, 1, 0, 0, 0]이 된다 broadCasting !
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invaild  = large_overlap & label_match # matching 되는 인덱스끼리 배열이 만들어진다. => Index Masking
            weights = detections[invaild, 4:5] # 1인 index만 켜져서 weights에 배열 상태로 저장된다. Weight란 => 4번쨰!!! Object Confidence이다.
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invaild, :4]).sum(0) / weights.sum() # object confidence와 detection의 값들이 곱해져서
            # 시그마[ obj_conf(여러개)*(det xywh) ] / 여러개 obj_conf 합
            # 그러니까 obj_conf로 weighted sum & norm을 해주는 것이다.
            keep_boxes += [detections[0]] # 가장 좋은 detection을 Keep 한다.
            detections = detections[~invaild] # 이제는 아까 invalid가 아니었던 애들 중에서 detection을 찾는다.
            #  그러면 다시 제일 score 높은게 위로간다. 그러면 이러한 제일 score가 높은것과 나머지를 비교한다. 그래서 iou가 threshold 보다 큰 것을 추려내고
        if keep_boxes: # 만약에 keep_boxes가 있다면 ->
            output[image_i] = torch.stack(keep_boxes)
            # torch.stack =>  [ torch.tensor, torch.tensor, ... ] 이렇게 구성되었다면 이것들을 torch.tensor 형태로 만드는 것이다.
            # 리스트에 torch.tensor 가 여러개 담겼다면... 이것들을 텐서형태로 만들자
            # 즉 이미지 image_i 인덱스에 대한 바운드 박스 몇개를 뽑아내는 것이다.
        ##################################################

    return output

def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0) # 값으로 0을 채우운다~
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1) # 1을 채우운다.
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors ])
    best_ious, best_n = ious.max(0)
    

# >> Test Phase
if __name__ == "__main__" :
    #classes_defs=load_classes("../data/coco.names")
    #print(classes_defs)
    a=torch.tensor( np.array([[20,20,40,40], [30,30,50,50]]), dtype=torch.float32)
    b=torch.tensor( np.array([[30,30,50,50], [40,40,70,70]]), dtype=torch.float32)
    print(a[:,0], b.shape)
    print(bbox_iou(a,b))

