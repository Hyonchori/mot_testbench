from typing import List
import numpy as np
import torch
import cv2


def preprocess(
        inputs: np.ndarray,  # [batch_size, height, width, channels]
        input_size: List[int],  # [height, width]
        device: torch.device,
        param_for_type  # half or not
):
    # unsqueeze array: [height, width, channels] -> [batch(1), height, width, channels]
    if isinstance(inputs, np.ndarray) and len(inputs.shape) == 3:
        inputs = inputs[None]

    # resize and pad
    if input_size is not None:
        imgs = [cv2.resize(img, dsize=(input_size[1], input_size[0])) for img in inputs]
        inputs = np.asarray(imgs)

    # list to np.ndarray
    if isinstance(inputs, list):
        inputs = np.asarray(inputs, np.float32)

    # float32 data type
    if inputs.dtype != np.float32:
        inputs = inputs.astype(np.float32)

    # bgr2rgb
    inputs = inputs[..., ::-1]

    # swap channels [batch_size, height, width, channels] -> [batch_size, channels, height, width]
    inputs = inputs.transpose(0, 3, 1, 2)

    # make contiguous array for faster inference
    inputs = np.ascontiguousarray(inputs)

    # numpy ndarray to tensor
    inputs = torch.from_numpy(inputs).to(device).type_as(param_for_type)

    return inputs
