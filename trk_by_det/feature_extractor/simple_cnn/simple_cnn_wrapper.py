# BringUp Simple_CNN for extracting feature (not for train)

import os
from typing import List
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from .model import Net
from .preprocessing import preprocess

FILE = Path(__file__).absolute()


class WrappedSimpleCNN(nn.Module):
    def __init__(
            self,
            num_classes: int = 751,  # number of classes in MARS dataset
            weights_file: str = None,
            input_size: List[int] = (128, 64),  # [height, width]
            device: torch.device = None
    ):
        super().__init__()
        print('\nLoading feature_extractor "SimpleCNN"...')

        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.num_classes = num_classes
        self.input_size = input_size

        self.model = self._init_model(num_classes, weights_file).to(self.device)
        self.model.eval()

        self.norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @staticmethod
    def _init_model(num_classes: int, weights_file: str) -> nn.Module:
        model = Net(num_classes=num_classes, reid=True)

        if weights_file is not None:
            if not os.path.isfile(weights_file):
                weights_dir = FILE.parents[0]
                weights_file = os.path.join(weights_dir, 'weights', weights_file)
            weights = torch.load(weights_file)['net_dict']
            model.load_state_dict(weights)
            print(f'\tpretrained extractor weights "{os.path.basename(weights_file)}" are loaded!')
        else:
            print('\tpretrained extractor weights is None.')

        return model

    def _preprocessing(self, inputs: np.ndarray):
        inputs = preprocess(inputs, self.input_size, self.norm_mean, self.norm_std, self.device)
        return inputs

    def forward(self, x: np.ndarray):
        x = self._preprocessing(x)
        x = self.model(x)
        x = x.cpu().data.numpy()
        return x


def get_simple_cnn_extractor(extractor_cfg, device: torch.device = None):
    return WrappedSimpleCNN(
        num_classes=751,
        weights_file='ckpt.t7',
        input_size=[128, 64] if extractor_cfg.extractor_input_size is None else extractor_cfg.extractor_input_size,
        device=device
    )
