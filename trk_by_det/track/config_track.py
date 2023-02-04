import math
import numpy as np

from .base_track import BaseTrack
from ..detection.base_detection import BaseDetection
from ..kalman_filter.base_kalman_filter import BaseKalmanFilter


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

    if cfg.type_feature == 'gallery':
        def feature_update_func(features: list, feature: np.ndarray, ema_alpha: float = None):
            features.append(feature)
            return features

    elif cfg.type_feature == 'ema':
        def feature_update_func(features: list, feature: np.ndarray, ema_alpha: float):
            smooth_feat = ema_alpha * features[-1] + (1 - ema_alpha) * feature
            smooth_feat /= np.linalg.norm(smooth_feat)
            features[-1] = smooth_feat
            return features

    elif cfg.type_feature is None:
        def feature_update_func(features: list, feature: np.ndarray, ema_alpha: float = None):
            return features

    else:
        def feature_update_func(features: list, feature: np.ndarray, ema_alpha: float = None):
            return features

    track_config['measure2xyxy_func'] = measure2xyxy
    track_config['feature_update_func'] = feature_update_func


def get_track(cfg, track_id: int, detection: BaseDetection, kalman_filter: BaseKalmanFilter):
    return BaseTrack(
        track_id=track_id,
        detection=detection,
        kalman_filter=kalman_filter,
        measure2xyxy_func=track_config['measure2xyxy_func'],
        feature_update_func=track_config['feature_update_func'],
        max_age=cfg.max_age,
        init_age=cfg.init_age,
        aspect_ratio_thr=cfg.aspect_ratio_thr,
        area_thr=cfg.area_thr,
        feature_gallery_len=cfg.feature_gallery_len,
        ema_alpha=cfg.ema_alpha,
        time_difference=cfg.time_difference,
    )
