from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_suppression
import matplotlib.pyplot as plt
import matplotlib.patches as patches
'''
module_i = "test"
>>> f"conv_{module_i}"
"conv_test" # 이렇게 나온다.
'''
def create_modules(module_defs):
    """
    모듈 리스트를 만든다. list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])] # 채널 개수
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            # 이것은 input = output 동일하게 하기 위한 pad size
            modules.add_module(
            f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn, # bn을 쓰면 bias를 안준다는 것이다.
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] =="maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["size"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0,1,0,1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

# output_filters >> [3, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 256, 512, 255, (255)] 이거를 [1:] 를 하니까 3은 빠진다. 그래서 16, 16~  ..., 4, 256, 512, 255, (255)] 이다.
# output_filters[module_i] -> # Out[4]: 255으로 나온다면 output_filters[1:][-4] 를 해버리면 가장 끝의 255가 -1인 기준으로 256 의 filter 수가 가져와진다.
        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shorcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            # Define detection Layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            # 여기서 [yolo] 를 읽고 anchor와 num_classes, img_size 대입( => img_size = int(hyperparams["height"]))
            modules.add_module(f"yolo_{module_i}", yolo_layer)

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list

class Upsample(nn.Module):
    """
    nn.Upsample is deprecated 됬다는?...
    """
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):

        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

# Input? output?
class YOLOLayer(nn.Module):
    """ Detection layer """
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        # BCELoss는 Sigmoid를 마지막단에 쓰는 경우 Binary Classification으로 쓰는 로스이다.
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0
        self.mse_loss = nn.MSELoss() # Multilabel Classification
        self.bce_loss = nn.BCELoss() # Binary     Classification

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size # grid_size가 13 정두
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size  # 416 / 13 => 32 인듯
        # torch.arange(g).repeat(g, 1) : [0,1,2,3,4,... g] 이렇게 아래 방향으로 g개 추가. 그리고 grid_y는 transpose이다.
        # ex : g=5 라면
        """
        grid_x = tensor([[[[0., 1., 2., 3., 4.],
          [0., 1., 2., 3., 4.],
          [0., 1., 2., 3., 4.],
          [0., 1., 2., 3., 4.],
          [0., 1., 2., 3., 4.]]]], device='cuda:0')
        grid_y = tensor([[[[0., 0., 0., 0., 0.],
          [1., 1., 1., 1., 1.],
          [2., 2., 2., 2., 2.],
          [3., 3., 3., 3., 3.],
          [4., 4., 4., 4., 4.]]]], device='cuda:0')
        """
        # repeat 시 차원이 하나 늘어난다.
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).to().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([( a_w / self.stride,
                                             a_h / self.stride) for a_w, a_h in self.anchors ])
        # anchors 사이즈를 stride 만큼 나눠서 줄인다. 이 경우 416/13 => 32 되서 32씩만큼
        # 기존 앵커 사이즈에서 나눠진다.
        # self.scaled_anchors.shape
        # >> (anchors_개수, 2(x,y))
        # self.anchor_w.shape = (1, anchors_개수, 1, 1)
        # self.anchor_h.shape = (1, anchors_개수, 1, 1)
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))


    def forward(self, x, targets=None, img_dim=None):

        # Tensors for cuda Support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0) # x.size() => torch.Size([1, 3, 13, 13])
        grid_size = x.size(2)   # 13, 13

        prediction = ( # (  )이렇게 감 쌈  그냥 가독성을 위한 것인듯... shape 모양에는 그대로임
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2) # 내부의 차원의 배치를 바꿀 것이다.
            .contigous() # 메모리를 연속적으로 할당해준다. 이렇게 하믄 backend에서 효율적으로 동작한다는듯
        )
        # ( num_samples, self.num_anchors, grid_size, grid_size, self.num_classes + 5 )
        # 만약 coco라면 => (1, 3, 13, 13, 85)

        # Get outputs
        x = torch.sigmoid(prediction[..., 0]) # => O(tx)이다. 즉, Sigmoid를 씌운 x 좌표
        y = torch.sigmoid(prediction[..., 1]) # => O(ty)이다. 즉, Sigmoid를 씌운 y 좌표
        w = prediction[..., 2]
        h = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls  = torch.sigmoid(prediction[..., 5:])

        # if grid size does not match current we compute new offsets
        #  맨처음에는 grid_size가 0이니까 if 안으로 빠진다.
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        # x.data.shape      -> [1, 3, 13, 13] 이다.
        # self.grid_x.shape -> [1, 1, 13, 13] 이다.
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w #
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h #
        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None :
            return output, 0
        else :
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th , tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet(nn.Module):
    """ YOLO v3 object detection model """
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0],"metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0],dtype=np.int32) #? 이건 뭐어지
        print("complete")

    # targets = None 이면 yolo_outputs 만, 그렇지 않으면 loss와 yolo_outputs 같이
    # Darknet이라는 모델의 중간 중간에 YoloLayer가 있다.
    # backbone 네트워크를 거치고 YoloLayer에서 찾는다. 근데 [yolo]가 여러개고 작은 yolo, 큰 yolo 에서 각각 feature가 뽑히고 그게 concat되어 합쳐진다.
    # 그것이 yolo_output
    def forward(self, x, targets=None):   # >> 여기로 이미지가 들어온다.
        img_dim = x.shape[2] # // => 416, 416 이라면 (batch_size, 3, 416, 416) 이렇게 들어가서 416이 img_dim이 된다.
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]: # 포함되면 ㅋ
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
                # bound 박스가 yolo detector에 의해서 torch.Size([1, 50, 85]) 이러한 형태로 뽑힘
                # [yolo]가 2개라면 torch.Size([1, 50, 85]), torch.Size([1, 300, 85]) 이런식으로 총 두번 이쪽으로 돈다.
            layer_outputs.append(x) # layer_ouput을 일단 여기다가 일단은 넣고 route나 shortcut에서 쓴다.
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1)) # 그게 여기서 다 합쳐져서 torch.Size([1, 350, 85]) > 이렇게 된다.
        return yolo_outputs if targets is None else (loss, yolo_outputs)


    def load_darknet_weights(self, weights_path):
        """ Parses and loads the weights stored in 'weights_path' """
        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header
            self.seen = header[3] # yolov3-tiny.weights를 까보니까 32013312 가 들어있음..
            weights = np.fromfile(f, dtype=np.float32) # weights는 개많음

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darkent53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        # module_def 와 module을 하나씩 받아서 weight를 로드한다.
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)): # 이미 > create_modules를 해서 얻어온 것
            if i == cutoff:
                break
            if module_def["type"] == "convolutional": #convoutional 한 부분만 안에 돌아가서 weight를 load한다. 
                conv_layer = module[0] # module[0] 이 conv 부분, module[1]이 batch_norm_6 부분, module[2]이 LeakyReLU 부분
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel() # output filter 만큼 bias 갯수가 붙겠네
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b) # 앞에서 bias만큼 뽑아낸당
                    ptr += num_b # bias만큼 뽑아낸 개수만큼 ptr이 증가된다.
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else :
                    # Load conv, bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b

                # Load conv, weights => conv_w는 weight 크기 -> 처음엔 16,3,3,3 -> 두번째는 32,16,3,3 -> 세번째는 64,32,3,3 -> 네번쨰는 128,64,3,3 -> .... 반복한다.
                num_w = conv_layer.weight.numel() # element의 총 갯수를 리턴해준다.
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                # ptr ~ ptr+num_w 까지 -> conv_layer.weight 의 형태 고대로 conv_w 만드는 것이다.
                conv_layer.weight.data.copy_(conv_w)  # 결과적으로 module[0].weight.data.copy_(내가 불러온 파일내용) => 이렇게 되어서 집어넣어지는 것
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
        > path - path of the new weights file,
        > cutoff - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[: cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # if batch norm, load bn first => 그건 cfg에서 batch_norm이 맨 앞쪽 부분에 있으니까
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp) # fp 라는 파일로 쓰겠다.
                    bn_layer.weight.data.cpu().numpy().tofile(fp) # fp 라는 파일로 쓰겠다.
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp) # fp 라는 파일로 쓰겠다.
                    bn_layer.running_var.data.cpu().numpy().tofile(fp) # fp 라는 파일로 쓰겠다.
                # Load conv bias
                else :
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()


if __name__ == "__main__" :
    model = Darknet("config/yolov3-tiny.cfg") # Darknet Init Test
    # 여기서 darknet weights를 load 할 수 있어야 하고
    # checkpoint weights를 load 할 수 있어야 한다.
    model.load_darknet_weights("weights/yolov3-tiny.weights") # weight load Setting
    
    print("멍멍")
