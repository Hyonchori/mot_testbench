# BringUp Fast_ReID for extracting feature (not for train)

import os
import sys
from typing import List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

FILE = Path(__file__).absolute()
sys.path.append(str(FILE.parents[1]))

from fast_reid.fastreid.config import get_cfg
from fast_reid.fastreid.modeling.meta_arch import build_model
from fast_reid.fastreid.utils.checkpoint import Checkpointer
from .preprocessing import preprocess


def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.freeze()
    return cfg


class WrappedFastReID(nn.Module):
    def __init__(
            self,
            config_file: str,
            weights_file: str = None,
            input_size: List[int] = (384, 128),  # [height, width]
            device: torch.device = None
    ):
        super().__init__()
        print('\nLoading feature_extractor "Fast-ReID"...')

        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if not os.path.isfile(config_file):
            cfg_dir = FILE.parents[0]
            config_file = os.path.join(cfg_dir, config_file)
        self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_file])
        self.input_size = input_size if input_size is not None else self.cfg.INPUT.SIZE_TEST

        self.model = self._init_model(self.cfg, weights_file)
        self.model.eval()
        if self.device != 'cpu':
            self.model = self.model.half()
            print('\tHalf tensor type!')

        print(next(self.model.parameters()).shape)

    @staticmethod
    def _init_model(config, weights_file: str) -> nn.Module:
        model = build_model(config)

        if weights_file is not None:
            if not os.path.isfile(weights_file):
                weights_dir = FILE.parents[0]
                weights_file = os.path.join(weights_dir, 'weights', weights_file)
            Checkpointer(model).load(weights_file)
            print(f'\tpretrained extractor weights "{os.path.basename(weights_file)}" are loaded!')
        else:
            print('\tpretrained extractor weights is None.')

        return model

    def _preprocessing(self, inputs: np.ndarray):
        inputs = preprocess(inputs, self.input_size, self.device, next(self.model.parameters()))
        return inputs

    def forward(self, x):
        x = self._preprocessing(x)
        x = self.model(x)
        x = F.normalize(x)  # Normalize feature to compute cosine distance
        x = x.cpu().data.numpy()
        return x


def get_fast_reid_extractor(extractor_cfg, device: torch.device = None):
    return WrappedFastReID(
        config_file='configs/MOT17/sbs_S50.yml',
        weights_file='mot17_sbs_S50.pth',
        input_size=extractor_cfg.extractor_input_size,
        device=device
    )


if __name__ == '__main__':
    cfg_file = 'configs/MOT17/sbs_S50.yml'
    wgt_file = 'mot17_sbs_S50.pth'
    reid = WrappedFastReID(cfg_file, wgt_file)
    pass
