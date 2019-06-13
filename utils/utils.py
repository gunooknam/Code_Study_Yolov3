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
    # -1까�? ?�는 ?�유 마�?막에 공백???�다.
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
    orig_h, orig_w = original_shape # ?�래??shape???�는??
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
# x,y??중심??좌표?�다. w,h??width?� height?�다. 
# xyxy => # left top, right bottom
def xywh2xyxy(x): # shape ?��??�이 가???�쪽??format??변�?
    y = x.new(x.shape) # torch??tensor?�는 new?�는 ?�수가 ?�음 그냥 x.shape?�도�?tensor 만드????
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
    # argsort??index ?�으�?sort ?�는 �?L=[5,2,3,5,6] # >>> np.argsort(L) # array([1, 2, 0, 3, 4], dtype=int64)
    # np.argsort(L) ?�에 L?�라??것에?��? -�?붙이�???�� 출력
    i = np.argsort(-conf)
    tp ,conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls) # True Object???�래?�이??

    # Create Precision-Recall curve and compute Ap for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum() # Number of ground truth objects
        n_p = i.sum() # Number of predicted objects


def compute_ap(recall, precision):
    '''
    # Args
        recall : The recall curve (list)   # recall�?precision??index??구간 개수??같을 것이?? 
        precision : The precision curve (list)
    # Returns
        The average precision as computed in py-faster-rcnn.
    '''
    # correct AP calculation
    # first append sentinel values at the end
    # 가로축?� Recall ?�고
    # ?�로축�? Precision ?�길 ?�한??
    mrec = np.concatenate(([0.0], recall, [1.0])) # recall ??관??curve List?�다.
    mpre = np.concatenate(([0,0], recall, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size -1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # ?�수 부????��
    # ex). mpre.size 10?�면 10, 9 ,8, 7, 6, 5, 4, 3, 2,...

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0] # ?�른 부분의 index�?리턴

    # and sum(\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i+1] ) # i???�덱?�이�??�문??
    # ?��?분�? ?�분 같�? ?�낌?�다.
    # mrec[i+1] - mrec[i]??구간 ?�이값들??리스?�이??
    # ?�것??마찬가지�?구간 ?�이값인 mpre[i+1]?�과 곱하???�것???�을 구하??�?=> AUC??
    return ap

def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = [] 
    for sample_i in range(len(outputs)): # output ?� 배열?�태

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
    wh2 = wh2.t() # torch.t()??그냥 transpose??
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h2 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    => torch Tensor ?�?�이 ?�으�??�어?�다. ??type?� forch.float64 or float32
    Args : box1???��?지 ?�?�이 ?�다.   x1, y1, x2, y2
    Case1 : box1[Num of Batch??개수,   0~3 ]?�라�? => 기존 ?�태�??��?
                                        xc, yc, w, h
    Case2 : box1[Num of Batch??개수,   0~3 ]?�라�? => x1y1x2y2�?바꾼??
    그래??IOU�?구할 ?�는 left Top, right bottom????좌표 ?�태�?바꿔???�다.
    """
    if not x1y1x2y2: # 만약??=> xc,yc,w,h (xc,yc?� ?�터??, ?�걸 left_top, right_bottom ?�로 바꿈
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else : # ?��? left_top, right_bottom ?�로 ?�어 ?�다�?
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
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1) # batch ?�으�??�번????구함
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1) # batch ?�으�??�번????구함
    print(b1_area, b2_area, inter_area)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou # batch 만큼??iou가 ?�긴 List


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    # ?�건 ??맥시�??�프?�션?��? 기법?�데 ?�떠??threshold 값을 ?�하�?
    ?�러??threshold 보다 confidence score가 ??���??�이 ???�보군에??지?�버리는 것이??
    Returns dectections with shape:
        (x1, y1, x2, y2, obejct_conf, class_score, class_pred)
    """
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    # prediction??batch?�위�??�을 것이??
    prediction[..., :4] = xywh2xyxy(prediction[..., :4]) # 코코�??�거 shape??(1,개많??85) ?�정???�다.
    # ?��?분에??iou 구하�??�한 top left, right bottom ?�식?�로 prediction?�라??tensor�?바꾸??부분이??

    output = [None for _ in range(len(prediction))] # => [None, None, None, None, None, None, None.... ]
    for image_i, image_pred in enumerate(prediction): # image index, image_pred 값을 뽑아?�당
        image_pred = image_pred[ image_pred[:,4] >= conf_thres ] # ?�기??개쪼�?추려진다. [10647, 85]-> [5,85] ?�도?
        # image_pred[:,4] ?�는 confidence 값이 conf_thres보다 ??경우??index�?추려?�다.
        #  -> �?값을 가�?image_pred가 ?�시 추려?�진??
        # image_pred??threshold???�해??걸러�?�?���?뽑아진다.!!

        if not image_pred.size(0): # 그런??만약??size??첫번�?rank가 0?�라�?..
            continue

        # Object Confidence times class confidence
        # Object Confidence?� Class Confidence?�???�른 ?�기??
        # Object Confidence??물체가 ?�기 Grid???�을 �?
        # Class  Confidence???�떤 ?�래?�에 ?�???�률?�다.
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # ?�덱?�마??가????class confidence�?뽑고 그것�?같�? index??object confidence?� 곱한??
        # max(1) ?�렇�??�면 2�??�상??tensor?�이 ?�옴 그중??0번째(값들)�?고르??것이??, 1번�?(index)??!

        #Sort By it             # ??버전?�서??pytorch?�서 argsort가 ?�되?��???..
        image_pred = image_pred[(-score).numpy().argsort()] # threshold�??�해??5개�? 추려졌다�??�걸 ?�렬?�서 배치?�다.
        # 그니�?index�?배열???�어�?그러�?�?배열??맞는 것만 추려�?
        # score가 ???��?�??�덱?��? 뽑아??> argsort() -> 값순?�로 ?�렬조건?��?�?index가 ?�렬?�다.
        # �?index??맞는 것을 ?�시 image_pred???�으므�?score ?�림 차순?�로 image_pred가 ??배열?�다.


                                            # 5~85�?까�???class confidence 중에??max�?뽑아?�다.
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)  # max�?찾고 rank ?�나가 감소?�다. But dimension???��??�는 Trick???�다.
        # threshold�??�해??5개�? 추려졌다�?5개의 confidence???�??값들 max 추린�?
        # threshold�??�해??5개�? 추려졌다�?몇번�?predict???�래?�인지??class index

                            # Cat???�는 것이?? xywh,class_confidence, class_label ?�렇�?추려진다.
        detections = torch.cat((image_pred[:, :5],
                                class_confs.float(),  # class_index???�당?�는 confidence가 뭐니?
                                class_preds.float()), # 몇번??class index??
                                1)
                                # 마�?�??�자??dimension?�데 ?�디�??�쳐줄�? ?�한??
                                # 1?????�유??=> ?�의 batch???�기??것을 ?�한 것이??

        # 그래??결과??threshold�?5개�? 추려졌다�?(5,7) ??tensor가 detection?�로 뽑아?��???것이??

        ######## Perform non-maximum suppression #########
        keep_boxes = []
        while detections.size(0): # 첫번�?score가 ?��? 것과 [1,4] -> ?�것�?   ?�전??추려??[5,4]�???비교?�다. 그래??[5] ?�는 iou가 ?�온?? => broadcast가 ?�는가 ?�다.
            large_overlap=bbox_iou( detections[0, :4].unsqueeze(0), detections[:, :4])  > nms_thres # Non Maximum Suppression Parameter
            label_match = detections[0, -1] == detections[:, -1] # 그래??score가 ?��? 것과 ?�머지 것들�??�벨??같�? 것을 추려?�다.
            # ?��? ?�어 detections[0, -1] ?� tensor(0.)
            #    그리�?detections[:, -1] ?� tensor([ 0.,  0., 17., 16., 17.]) ?�고 ?�자 그러�?매치 ?�는 것�? [1, 1, 0, 0, 0]???�다 broadCasting !
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invaild  = large_overlap & label_match # matching ?�는 ?�덱?�끼�?배열??만들?�진?? => Index Masking
            weights = detections[invaild, 4:5] # 1??index�?켜져??weights??배열 ?�태�??�?�된?? Weight?� => 4번�?!!! Object Confidence?�다.
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invaild, :4]).sum(0) / weights.sum() # object confidence?� detection??값들??곱해?�서
            # ?�그�? obj_conf(?�러�?*(det xywh) ] / ?�러�?obj_conf ??
            # 그러?�까 obj_conf�?weighted sum & norm???�주??것이??
            keep_boxes += [detections[0]] # 가??좋�? detection??Keep ?�다.
            detections = detections[~invaild] # ?�제???�까 invalid가 ?�니?�던 ?�들 중에??detection??찾는??
            #  그러�??�시 ?�일 score ?��?�??�로간다. 그러�??�러???�일 score가 ?��?것과 ?�머지�?비교?�다. 그래??iou가 threshold 보다 ??것을 추려?�고
        if keep_boxes: # 만약??keep_boxes가 ?�다�?->
            output[image_i] = torch.stack(keep_boxes)
            # torch.stack =>  [ torch.tensor, torch.tensor, ... ] ?�렇�?구성?�었?�면 ?�것?�을 torch.tensor ?�태�?만드??것이??
            # 리스?�에 torch.tensor 가 ?�러�??�겼?�면... ?�것?�을 ?�서?�태�?만들??
            # �??��?지 image_i ?�덱?�에 ?�??바운??박스 몇개�?뽑아?�는 것이??
        ##################################################

    return output



# ?�주?�에
def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0) # 값으�?0??채우?�다~
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1) # 1??채우?�다.
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
    # Separate target Values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


# >> Test Phase
if __name__ == "__main__" :
    #classes_defs=load_classes("../data/coco.names")
    #print(classes_defs)
    a=torch.tensor( np.array([[20,20,40,40], [30,30,50,50]]), dtype=torch.float32)
    b=torch.tensor( np.array([[30,30,50,50], [40,40,70,70]]), dtype=torch.float32)
    print(a[:,0], b.shape)
    print(bbox_iou(a,b))

