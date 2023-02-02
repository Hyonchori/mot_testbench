import math
import numpy as np

from .base_track import BaseTrack


track_config = {}


def init_track_config(cfg):
    if cfg.type_state == 'cpsa':
        def measure2xyxy(z):
            cpsa = [x[0] for x in z]
            width = math.sqrt(max(1, cpsa[2] * cpsa[3]))
            height = max(1, cpsa[2] / width)
            xyxy = [
                cpsa[0] - width * 0.5,
                cpsa[1] - height * 0.5,
                cpsa[0] + width * 0.5,
                cpsa[1] + height * 0.5
            ]
            return np.asarray(xyxy)

    elif cfg.type_state == 'cpah':
        def measure2xyxy(z):
            cpah = [x[0] for x in z]
            width = max(1, cpah[2] * cpah[3])
            height = max(1, cpah[3])
            xyxy = [
                cpah[0] - width * 0.5,
                cpah[1] - height * 0.5,
                cpah[0] + width * 0.5,
                cpah[1] + height * 0.5,
            ]
            return np.asarray(xyxy)

    elif cfg.type_state == 'cpwh':
        def measure2xyxy(z):
            cpwh = [x[0] for x in z]
            width = max(1, cpwh[2])
            height = max(1, cpwh[3])
            xyxy = [
                cpwh[0] - width * 0.5,
                cpwh[1] - height * 0.5,
                cpwh[0] + width * 0.5,
                cpwh[1] + height * 0.5,
            ]
            return np.asarray(xyxy)

    else:
        def measure2xyxy(z):
            cpah = [x[0] for x in z]
            width = max(1, cpah[2] * cpah[3])
            height = max(1, cpah[3])
            xyxy = [
                cpah[0] - width * 0.5,
                cpah[1] - height * 0.5,
                cpah[0] + width * 0.5,
                cpah[1] + height * 0.5,
            ]
            return np.asarray(xyxy)

    track_config['measure2xyxy_func'] = measure2xyxy


