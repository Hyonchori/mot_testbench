# BringUp YOLOx for inference (not for train)

import os
from typing import List
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from .models.yolo_pafpn import YOLOPAFPN
from .models.yolo_head import YOLOXHead
from .models.yolox import YOLOX
from .preprocessing import preprocess
from .postprocessing import postprocess

FILE = Path(__file__).absolute()


class WrappedYOLOX(nn.Module):
    def __init__(
            self,
            depth: float = 1.0,
            width: float = 1.0,
            num_classes: int = 80,  # number of classes in COCO
            weights_file: str = None,
            input_size: List[int] = (720, 1280),  # [height, width]
            device: torch.device = None,
            conf_thr: float = 0.7,
            iou_thr: float = 0.5,
            fuse: bool = False,
            half: bool = False,
    ):
        super().__init__()
        print('\nLoading detector "YOLOX"...')
        print(f'\tconf_thr: {conf_thr}, iou_thr: {iou_thr}, img_size: {input_size}, fuse: {fuse}, half: {half}')
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.num_classes = num_classes
        self.input_size = input_size
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.half = half

        self.model = self._init_model(depth, width, num_classes, weights_file).to(self.device)
        self.model.eval()

        if fuse:
            self.model = fuse_model(self.model)
            print('\tFusing layer complete!')

        if half:
            self.model.half()
            print('\tHalf tensor type!')

        self.norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @staticmethod
    def _init_model(depth: float, width: float, num_classes: int, weights_file: str) -> nn.Module:
        def init_yolox(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
        head = YOLOXHead(num_classes, width, in_channels=in_channels)
        model = YOLOX(backbone, head)

        model.apply(init_yolox)
        model.head.initialize_biases(1e-2)

        if weights_file is not None:
            if not os.path.isfile(weights_file):
                weights_dir = Path(__file__).absolute().parents[0]
                weights_file = os.path.join(weights_dir, 'weights', weights_file)
            weights = torch.load(weights_file)['model']
            model.load_state_dict(weights)
            print(f'\tpretrained detector weights "{os.path.basename(weights_file)}" are loaded!')
        else:
            print('\tpretrained detector weights is None.')
        return model

    def _preprocessing(self, inputs: np.ndarray):
        inputs = preprocess(inputs, self.input_size, self.norm_mean, self.norm_std, self.device, self.half)
        return inputs

    def _postprocessing(self, outputs: torch.Tensor):
        outputs = postprocess(outputs, self.num_classes, self.conf_thr, self.iou_thr)
        return outputs

    def forward(self, x: np.ndarray):
        x = self._preprocessing(x)
        x = self.model(x)
        x = self._postprocessing(x)
        return x


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def fuse_model(model):
    from .models.network_blocks import BaseConv

    for m in model.modules():
        if type(m) is BaseConv and hasattr(m, "bn"):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, "bn")  # remove batchnorm
            m.forward = m.fuseforward  # update forward
    return model


def get_wrapped_yolox(det_cfg, device: torch.device = None):
    if det_cfg.detector_weights == 'yolox_x_coco':
        depth = 1.33
        width = 1.25
        num_classes = 80
        input_size = [800, 1440]
        weights_file = 'yolox_x.pth'

    elif det_cfg.detector_weights == 'yolox_m_byte_mot17':
        depth = 0.67
        width = 0.75
        num_classes = 1
        input_size = [800, 1440]
        weights_file = 'bytetrack_m_mot17.pth.tar'

    elif det_cfg.detector_weights == 'yolox_l_byte_mot17':
        depth = 1.0
        width = 1.0
        num_classes = 1
        input_size = [800, 1440]
        weights_file = 'bytetrack_l_mot17.pth.tar'

    elif det_cfg.detector_weights == 'yolox_x_byte_mot17':
        depth = 1.33
        width = 1.25
        num_classes = 1
        input_size = [800, 1440]
        weights_file = 'bytetrack_x_mot17.pth.tar'

    elif det_cfg.detector_weights == 'yolox_x_byte_mot20':
        depth = 1.33
        width = 1.25
        num_classes = 1
        input_size = [896, 1600]
        weights_file = 'bytetrack_x_mot20.tar'

    elif det_cfg.detector_weights == 'yolox_x_byte_ablation':
        depth = 1.33
        width = 1.25
        num_classes = 1
        input_size = [800, 1440]
        weights_file = 'bytetrack_ablation.pth.tar'

    elif det_cfg.detector_weights == 'yolox_x_ocsort_dance':
        depth = 1.33
        width = 1.25
        num_classes = 1
        input_size = [800, 1440]
        weights_file = 'ocsort_dance_model.pth.tar'

    else:
        raise Exception(f'Given detector weights "{det_cfg.detector_weights}" is not valid for wrapped_yolox!')

    return WrappedYOLOX(
        depth=depth,
        width=width,
        num_classes=num_classes,
        weights_file=weights_file,
        input_size=input_size if det_cfg.detector_input_size is None else det_cfg.detector_input_size,
        device=device,
        conf_thr=det_cfg.detector_conf_thr,
        iou_thr=det_cfg.detector_iou_thr,
        fuse=True,
        half=True
    )
