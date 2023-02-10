# BringUp SLM(Similarity Learning Module) for extracting feature (not for train)

import os
from typing import List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .networks_ver12 import DLASeg

FILE = Path(__file__).absolute()


class WrappedSLM(nn.Module):
    def __init__(
            self,
            weights_file: str = None,
            device: torch.device = None
    ):
        super().__init__()
        print('\nLoading feature_extractor "SLM"...')

        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = self._init_model(weights_file).to(self.device)
        self.model.eval()

    @staticmethod
    def _init_model(weights_file: str) -> nn.Module:
        model = DLASeg()

        if weights_file is not None:
            if not os.path.isfile(weights_file):
                weights_dir = FILE.parents[0]
                weights_file = os.path.join(weights_dir, 'weights', weights_file)

            checkpoint = torch.load(weights_file, map_location=lambda storage, loc: storage)
            state_dict_ = checkpoint['model_G_state_dict']
            state_dict = {}

            # convert data_parallal to model
            for k in state_dict_:
                if k.startswith('module') and not k.startswith('module_list'):
                    state_dict[k[7:]] = state_dict_[k]
                else:
                    state_dict[k] = state_dict_[k]
            model_state_dict = model.state_dict()

            # check loaded parameters and created model parameters
            msg = 'If you see this, your model does not fully load the ' + \
                  'pre-trained weight. Please make sure ' + \
                  'you have correctly specified --arch xxx ' + \
                  'or set the correct --num_classes for your own dataset.'
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, ' \
                              'loaded shape{}. {}'.format(
                            k, model_state_dict[k].shape, state_dict[k].shape, msg))
                        state_dict[k] = model_state_dict[k]
                else:
                    print('Drop parameter {}.'.format(k) + msg)
            for k in model_state_dict:
                if not (k in state_dict):
                    print('No param {}.'.format(k) + msg)
                    state_dict[k] = model_state_dict[k]
            model.load_state_dict(state_dict, strict=False)
            print(f'\tpretrained extractor weights "{os.path.basename(weights_file)}" are loaded!')
        else:
            print('\tpretrained extractor weights is None.')

        return model

    def forward(self, x: np.ndarray):
        features = torch.zeros((len(x), 128), dtype=torch.float64)

        for time in range(len(x)):
            x[time] = torch.from_numpy(x[time]).to(self.device)
            features[time, :] = self.model.inference_forward_fast(x[time].float())
        features = F.normalize(features)
        features_keep = features.cpu().numpy()
        return features_keep


def get_slm_extractor(extractor_cfg, device: torch.device = None):
    return WrappedSLM(
        weights_file='ver12.pt',
        device=device
    )
