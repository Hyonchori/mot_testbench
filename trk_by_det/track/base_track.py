import math
from collections import deque

import numpy as np

from ..detection.base_detection import BaseDetection
from ..kalman_filter.base_kalman_filter import BaseKalmanFilter


class TrackState:
    Tentative = 1
    Ambiguous = 2
    Confirmed = 3
    Rematched = 4
    Lost = 5


class BaseTrack:
    def __init__(
            self,
            track_id: int,
            detection: BaseDetection,
            kalman_filter: BaseKalmanFilter,
            measure2xyxy_func,
            max_age: int = 30,
            init_age: int = 3,
            feature_gallery_len: int = 100,
            ema_alpha: float = 0.9,
            time_difference: int = 3,
    ):
        self.track_id = track_id
        self.max_age = max_age
        self.init_age = init_age

        self.kf = kalman_filter
        self.conf = detection.conf
        self.measure2xyxy_func = measure2xyxy_func

        