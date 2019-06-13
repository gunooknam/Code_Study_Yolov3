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
    # -1ê¹Œì? ?˜ëŠ” ?´ìœ  ë§ˆì?ë§‰ì— ê³µë°±???ˆë‹¤.
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
    orig_h, orig_w = original_shape # ?ëž˜??shape???£ëŠ”??
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
# x,y??ì¤‘ì‹¬??ì¢Œí‘œ?´ë‹¤. w,h??width?€ height?´ë‹¤. 
# xyxy => # left top, right bottom
def xywh2xyxy(x): # shape ?ê??†ì´ ê°€???ìª½??format??ë³€ê²?
    y = x.new(x.shape) # torch??tensor?ëŠ” new?¼ëŠ” ?¨ìˆ˜ê°€ ?ˆìŒ ê·¸ëƒ¥ x.shape?•ë„ë¡?tensor ë§Œë“œ????
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
    # argsort??index ?œìœ¼ë¡?sort ?˜ëŠ” ê²?L=[5,2,3,5,6] # >>> np.argsort(L) # array([1, 2, 0, 3, 4], dtype=int64)
    # np.argsort(L) ?ˆì— L?´ë¼??ê²ƒì—?¤ê? -ë¥?ë¶™ì´ë©???ˆœ ì¶œë ¥
    i = np.argsort(-conf)
    tp ,conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls) # True Object???´ëž˜?¤ì´??

    # Create Precision-Recall curve and compute Ap for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum() # Number of ground truth objects
        n_p = i.sum() # Number of predicted objects


def compute_ap(recall, precision):
    '''
    # Args
        recall : The recall curve (list)   # recallê³?precision??index??êµ¬ê°„ ê°œìˆ˜??ê°™ì„ ê²ƒì´?? 
        precision : The precision curve (list)
    # Returns
        The average precision as computed in py-faster-rcnn.
    '''
    # correct AP calculation
    # first append sentinel values at the end
    # ê°€ë¡œì¶•?€ Recall ?´ê³ 
    # ?¸ë¡œì¶•ì? Precision ?´ê¸¸ ?í•œ??
    mrec = np.concatenate(([0.0], recall, [1.0])) # recall ??ê´€??curve List?´ë‹¤.
    mpre = np.concatenate(([0,0], recall, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size -1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # ?°ìˆ˜ ë¶€????ˆœ
    # ex). mpre.size 10?´ë©´ 10, 9 ,8, 7, 6, 5, 4, 3, 2,...

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0] # ?¤ë¥¸ ë¶€ë¶„ì˜ indexë¥?ë¦¬í„´

    # and sum(\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i+1] ) # i???¸ë±?¤ì´ê¸??Œë¬¸??
    # ?´ë?ë¶„ì? ?ë¶„ ê°™ì? ?ë‚Œ?´ë‹¤.
    # mrec[i+1] - mrec[i]??êµ¬ê°„ ?¬ì´ê°’ë“¤??ë¦¬ìŠ¤?¸ì´??
    # ?´ê²ƒ??ë§ˆì°¬ê°€ì§€ë¡?êµ¬ê°„ ?’ì´ê°’ì¸ mpre[i+1]?¤ê³¼ ê³±í•˜???´ê²ƒ???©ì„ êµ¬í•˜??ê²?=> AUC??
    return ap

def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = [] 
    for sample_i in range(len(outputs)): # output ?€ ë°°ì—´?•íƒœ

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
    wh2 = wh2.t() # torch.t()??ê·¸ëƒ¥ transpose??
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h2 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    => torch Tensor ?€?…ì´ ?ˆìœ¼ë¡??¤ì–´?¨ë‹¤. ??type?€ forch.float64 or float32
    Args : box1???ê?ì§€ ?€?…ì´ ?ˆë‹¤.   x1, y1, x2, y2
    Case1 : box1[Num of Batch??ê°œìˆ˜,   0~3 ]?´ë¼ë©? => ê¸°ì¡´ ?íƒœë¥?? ì?
                                        xc, yc, w, h
    Case2 : box1[Num of Batch??ê°œìˆ˜,   0~3 ]?´ë¼ë©? => x1y1x2y2ë¡?ë°”ê¾¼??
    ê·¸ëž˜??IOUë¥?êµ¬í•  ?ŒëŠ” left Top, right bottom????ì¢Œí‘œ ?•íƒœë¡?ë°”ê¿”???œë‹¤.
    """
    if not x1y1x2y2: # ë§Œì•½??=> xc,yc,w,h (xc,yc?€ ?¼í„°??, ?´ê±¸ left_top, right_bottom ?¼ë¡œ ë°”ê¿ˆ
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else : # ?´ë? left_top, right_bottom ?¼ë¡œ ?˜ì–´ ?ˆë‹¤ë©?
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
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1) # batch ?µìœ¼ë¡??œë²ˆ????êµ¬í•¨
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1) # batch ?µìœ¼ë¡??œë²ˆ????êµ¬í•¨
    print(b1_area, b2_area, inter_area)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou # batch ë§Œí¼??iouê°€ ?´ê¸´ List


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    # ?´ê±´ ??ë§¥ì‹œë©??œí”„?ˆì…˜?´ë? ê¸°ë²•?¸ë° ?´ë– ??threshold ê°’ì„ ?•í•˜ê³?
    ?´ëŸ¬??threshold ë³´ë‹¤ confidence scoreê°€ ??œ¼ë©??µì´ ???„ë³´êµ°ì—??ì§€?Œë²„ë¦¬ëŠ” ê²ƒì´??
    Returns dectections with shape:
        (x1, y1, x2, y2, obejct_conf, class_score, class_pred)
    """
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    # prediction??batch?¨ìœ„ë¡??ˆì„ ê²ƒì´??
    prediction[..., :4] = xywh2xyxy(prediction[..., :4]) # ì½”ì½”ë©??´ê±° shape??(1,ê°œë§Ž??85) ?´ì •???œë‹¤.
    # ?´ë?ë¶„ì—??iou êµ¬í•˜ê¸??„í•œ top left, right bottom ?•ì‹?¼ë¡œ prediction?´ë¼??tensorë¡?ë°”ê¾¸??ë¶€ë¶„ì´??

    output = [None for _ in range(len(prediction))] # => [None, None, None, None, None, None, None.... ]
    for image_i, image_pred in enumerate(prediction): # image index, image_pred ê°’ì„ ë½‘ì•„?¸ë‹¹
        image_pred = image_pred[ image_pred[:,4] >= conf_thres ] # ?”ê¸°??ê°œìª¼ê¸?ì¶”ë ¤ì§„ë‹¤. [10647, 85]-> [5,85] ?•ë„?
        # image_pred[:,4] ?¼ëŠ” confidence ê°’ì´ conf_thresë³´ë‹¤ ??ê²½ìš°??indexë¥?ì¶”ë ¤?¸ë‹¤.
        #  -> ê·?ê°’ì„ ê°€ì§?image_predê°€ ?¤ì‹œ ì¶”ë ¤?´ì§„??
        # image_pred??threshold???˜í•´??ê±¸ëŸ¬ì§?ê°?ˆ˜ë¡?ë½‘ì•„ì§„ë‹¤.!!

        if not image_pred.size(0): # ê·¸ëŸ°??ë§Œì•½??size??ì²«ë²ˆì§?rankê°€ 0?´ë¼ë¯?..
            continue

        # Object Confidence times class confidence
        # Object Confidence?€ Class Confidence?€???¤ë¥¸ ?˜ê¸°??
        # Object Confidence??ë¬¼ì²´ê°€ ?¬ê¸° Grid???ˆì„ ê¹?
        # Class  Confidence???´ë–¤ ?´ëž˜?¤ì— ?€???•ë¥ ?´ë‹¤.
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # ?¸ë±?¤ë§ˆ??ê°€????class confidenceë¥?ë½‘ê³  ê·¸ê²ƒê³?ê°™ì? index??object confidence?€ ê³±í•œ??
        # max(1) ?´ë ‡ê²??˜ë©´ 2ê°??´ìƒ??tensor?¤ì´ ?˜ì˜´ ê·¸ì¤‘??0ë²ˆì§¸(ê°’ë“¤)ë¥?ê³ ë¥´??ê²ƒì´??, 1ë²ˆì?(index)??!

        #Sort By it             # ??ë²„ì „?ì„œ??pytorch?ì„œ argsortê°€ ?ˆë˜?˜ë???..
        image_pred = image_pred[(-score).numpy().argsort()] # thresholdë¡??¸í•´??5ê°œê? ì¶”ë ¤ì¡Œë‹¤ë©??´ê±¸ ?•ë ¬?´ì„œ ë°°ì¹˜?œë‹¤.
        # ê·¸ë‹ˆê¹?indexë¡?ë°°ì—´???¤ì–´ê°?ê·¸ëŸ¬ë©?ê·?ë°°ì—´??ë§žëŠ” ê²ƒë§Œ ì¶”ë ¤ì§?
        # scoreê°€ ???œë?ë¡??¸ë±?¤ë? ë½‘ì•„??> argsort() -> ê°’ìˆœ?¼ë¡œ ?•ë ¬ì¡°ê±´?´ì?ë§?indexê°€ ?•ë ¬?œë‹¤.
        # ê·?index??ë§žëŠ” ê²ƒì„ ?¤ì‹œ image_pred???£ìœ¼ë¯€ë¡?score ?´ë¦¼ ì°¨ìˆœ?¼ë¡œ image_predê°€ ??ë°°ì—´?œë‹¤.


                                            # 5~85ê°?ê¹Œì???class confidence ì¤‘ì—??maxë¥?ë½‘ì•„?¸ë‹¤.
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)  # maxë¥?ì°¾ê³  rank ?˜ë‚˜ê°€ ê°ì†Œ?œë‹¤. But dimension??? ì??˜ëŠ” Trick???»ë‹¤.
        # thresholdë¡??¸í•´??5ê°œê? ì¶”ë ¤ì¡Œë‹¤ë©?5ê°œì˜ confidence???€??ê°’ë“¤ max ì¶”ë¦°ê²?
        # thresholdë¡??¸í•´??5ê°œê? ì¶”ë ¤ì¡Œë‹¤ë©?ëª‡ë²ˆì§?predict???´ëž˜?¤ì¸ì§€??class index

                            # Cat???˜ëŠ” ê²ƒì´?? xywh,class_confidence, class_label ?´ë ‡ê²?ì¶”ë ¤ì§„ë‹¤.
        detections = torch.cat((image_pred[:, :5],
                                class_confs.float(),  # class_index???´ë‹¹?˜ëŠ” confidenceê°€ ë­ë‹ˆ?
                                class_preds.float()), # ëª‡ë²ˆ??class index??
                                1)
                                # ë§ˆì?ë§??¸ìž??dimension?¸ë° ?´ë””ë¥??©ì³ì¤„ì? ?•í•œ??
                                # 1?????´ìœ ??=> ?žì˜ batch???¨ê¸°??ê²ƒì„ ?í•œ ê²ƒì´??

        # ê·¸ëž˜??ê²°ê³¼??thresholdë¡?5ê°œê? ì¶”ë ¤ì¡Œë‹¤ë©?(5,7) ??tensorê°€ detection?¼ë¡œ ë½‘ì•„?´ì???ê²ƒì´??

        ######## Perform non-maximum suppression #########
        keep_boxes = []
        while detections.size(0): # ì²«ë²ˆì§?scoreê°€ ?’ì? ê²ƒê³¼ [1,4] -> ?´ê²ƒê³?   ?´ì „??ì¶”ë ¤??[5,4]ë¥???ë¹„êµ?œë‹¤. ê·¸ëž˜??[5] ?¼ëŠ” iouê°€ ?˜ì˜¨?? => broadcastê°€ ?˜ëŠ”ê°€ ?¶ë‹¤.
            large_overlap=bbox_iou( detections[0, :4].unsqueeze(0), detections[:, :4])  > nms_thres # Non Maximum Suppression Parameter
            label_match = detections[0, -1] == detections[:, -1] # ê·¸ëž˜??scoreê°€ ?’ì? ê²ƒê³¼ ?˜ë¨¸ì§€ ê²ƒë“¤ê³??¼ë²¨??ê°™ì? ê²ƒì„ ì¶”ë ¤?¸ë‹¤.
            # ?ˆë? ?¤ì–´ detections[0, -1] ?€ tensor(0.)
            #    ê·¸ë¦¬ê³?detections[:, -1] ?€ tensor([ 0.,  0., 17., 16., 17.]) ?¼ê³  ?˜ìž ê·¸ëŸ¬ë©?ë§¤ì¹˜ ?˜ëŠ” ê²ƒì? [1, 1, 0, 0, 0]???œë‹¤ broadCasting !
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invaild  = large_overlap & label_match # matching ?˜ëŠ” ?¸ë±?¤ë¼ë¦?ë°°ì—´??ë§Œë“¤?´ì§„?? => Index Masking
            weights = detections[invaild, 4:5] # 1??indexë§?ì¼œì ¸??weights??ë°°ì—´ ?íƒœë¡??€?¥ëœ?? Weight?€ => 4ë²ˆì?!!! Object Confidence?´ë‹¤.
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invaild, :4]).sum(0) / weights.sum() # object confidence?€ detection??ê°’ë“¤??ê³±í•´?¸ì„œ
            # ?œê·¸ë§? obj_conf(?¬ëŸ¬ê°?*(det xywh) ] / ?¬ëŸ¬ê°?obj_conf ??
            # ê·¸ëŸ¬?ˆê¹Œ obj_confë¡?weighted sum & norm???´ì£¼??ê²ƒì´??
            keep_boxes += [detections[0]] # ê°€??ì¢‹ì? detection??Keep ?œë‹¤.
            detections = detections[~invaild] # ?´ì œ???„ê¹Œ invalidê°€ ?„ë‹ˆ?ˆë˜ ? ë“¤ ì¤‘ì—??detection??ì°¾ëŠ”??
            #  ê·¸ëŸ¬ë©??¤ì‹œ ?œì¼ score ?’ì?ê²??„ë¡œê°„ë‹¤. ê·¸ëŸ¬ë©??´ëŸ¬???œì¼ scoreê°€ ?’ì?ê²ƒê³¼ ?˜ë¨¸ì§€ë¥?ë¹„êµ?œë‹¤. ê·¸ëž˜??iouê°€ threshold ë³´ë‹¤ ??ê²ƒì„ ì¶”ë ¤?´ê³ 
        if keep_boxes: # ë§Œì•½??keep_boxesê°€ ?ˆë‹¤ë©?->
            output[image_i] = torch.stack(keep_boxes)
            # torch.stack =>  [ torch.tensor, torch.tensor, ... ] ?´ë ‡ê²?êµ¬ì„±?˜ì—ˆ?¤ë©´ ?´ê²ƒ?¤ì„ torch.tensor ?•íƒœë¡?ë§Œë“œ??ê²ƒì´??
            # ë¦¬ìŠ¤?¸ì— torch.tensor ê°€ ?¬ëŸ¬ê°??´ê²¼?¤ë©´... ?´ê²ƒ?¤ì„ ?ì„œ?•íƒœë¡?ë§Œë“¤??
            # ì¦??´ë?ì§€ image_i ?¸ë±?¤ì— ?€??ë°”ìš´??ë°•ìŠ¤ ëª‡ê°œë¥?ë½‘ì•„?´ëŠ” ê²ƒì´??
        ##################################################

    return output



# ?˜ì£¼?…ì—
def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0) # ê°’ìœ¼ë¡?0??ì±„ìš°?´ë‹¤~
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1) # 1??ì±„ìš°?´ë‹¤.
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

