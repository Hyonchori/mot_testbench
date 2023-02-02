import math
from typing import List

import cv2
import torch
import numpy as np


def letterbox(
        img: np.ndarray,
        new_shape=(640, 640),
        color: int = (114, 114, 114),
        auto: bool = True,
        stretch: bool = False,
        stride: int = 32,
        dnn_pad: bool = False
):
    # resize and pad image while meeting stride-multiple constraints
    shape = img.shape[: 2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    if dnn_pad:
        new_shape = [stride * math.ceil(x / stride) for x in new_shape]

    if img.shape[:2] == new_shape:
        return img, 1., (0, 0)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif stretch:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return img, ratio, (dw, dh)


def preprocess(
        inputs: np.ndarray,  # [batch_size, height, width, channels]
        input_size: List[int],  # [height, width]
        norm_mean: np.ndarray,
        norm_std: np.ndarray,
        device: torch.device,
        half: bool
):
    # unsqueeze array: [height, width, channels] -> [batch(1), height, width, channels]
    if isinstance(inputs, np.ndarray) and len(inputs.shape) == 3:
        inputs = inputs[None]

    # resize and pad
    if input_size is not None:
        imgs = [letterbox(img, input_size, auto=False, dnn_pad=True)[0] for img in inputs]
        inputs = np.asarray(imgs)

    # list to np.ndarray
    if isinstance(inputs, list):
        inputs = np.asarray(inputs, np.float32)

    # float32 data type
    if inputs.dtype != np.float32:
        inputs = inputs.astype(np.float32)

    # scaling: [0 ~ 255] -> [0.0 ~ 1.0]
    inputs /= 255.

    # normalize
    inputs -= norm_mean
    inputs /= norm_std

    # bgr2rgb
    inputs = inputs[..., ::-1]

    # swap channels [batch_size, height, width, channels] -> [batch_size, channels, height, width]
    inputs = inputs.transpose(0, 3, 1, 2)

    # make contiguous array for faster inference
    inputs = np.ascontiguousarray(inputs)

    # numpy ndarray to tensor
    inputs = torch.from_numpy(inputs).to(device)

    # half tensor type for faster inference
    if half:
        inputs = inputs.half()
    return inputs
