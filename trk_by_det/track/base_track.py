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
            feature_update_func,
            max_age: int = 30,
            init_age: int = 3,
            aspect_ratio_thr: float = 1.6,
            area_thr: int = 100,
            feature_gallery_len: int = 100,
            ema_alpha: float = 0.9,
            time_difference: int = 3,
            apply_obs_to_lost: bool = True
    ):
        self.track_id = track_id
        self.max_age = max_age
        self.init_age = init_age
        self.aspect_ratio_thr = aspect_ratio_thr
        self.area_thr = area_thr

        self.kf = kalman_filter
        self.conf = detection.conf
        self.measure2xyxy_func = measure2xyxy_func

        self.features = deque([detection.feature], maxlen=feature_gallery_len)
        self.feature_update_func = feature_update_func
        self.ema_alpha = ema_alpha
        self.time_difference = time_difference
        self.apply_obs_to_lost = apply_obs_to_lost

        self.x, self.x_cov = self.get_state()
        self.is_matched = False
        self.age = 0
        self.hits = 0
        self.time_since_update = 0
        self.last_observation = detection.z
        self.observations = {0: self.last_observation}
        self.velocity = np.array([0, 0])
        self.direction = np.array([0, 0])
        self.speed = 0.0
        self.track_state = TrackState.Tentative

    def get_state(self):
        return self.kf.x, self.kf.x_cov

    def get_projected_state(self):
        return self.kf.project()

    def predict(self):
        self.is_matched = False
        self.age += 1
        self.time_since_update += 1
        if self.apply_obs_to_lost and self.is_lost():
            tmp_z = self.last_observation.copy()
            tmp_z[:2, 0] += self.velocity * (self.time_since_update - 1)
            self.kf.z = tmp_z
            self.x, self.x_cov = self.kf.initialize_state()
            self.kf.x = self.x
            self.kf.x_cov = self.x_cov
            # self.kf.predict()
            # self.x, self.x_cov = self.get_state()
        else:
            self.kf.predict()
            self.x, self.x_cov = self.get_state()

    def measure(self, detection: BaseDetection, apply_oos: bool = False):
        self.is_matched = True
        self.kf.measure(detection.z.copy(), detection.conf, apply_oos, self.time_since_update)
        self.conf = detection.conf

        # update feature
        self.features = self.feature_update_func(self.features, detection.feature, self.ema_alpha)

        # update velocity
        previous_obs, dt = self.get_previous_observation()
        tmp_obs = detection.z
        if previous_obs is not None:
            self.velocity, self.direction, self.speed = self.get_speed(previous_obs[:2, 0], tmp_obs[:2, 0], dt)
        self.last_observation = tmp_obs
        self.observations[self.age] = tmp_obs

    def get_previous_observation(self):
        previous_center = None
        dt = None
        for i in range(self.time_difference):
            dt = self.time_difference - i
            if self.age - dt in self.observations:
                previous_center = self.observations[self.age - dt]
                break
        if previous_center is None:
            previous_center = self.last_observation
        return previous_center, dt

    def get_speed(self, center1: np.ndarray, center2: np.ndarray, dt: int):
        velocity = center2 - center1
        norm = np.linalg.norm(velocity) + 1e-6
        return velocity / dt, velocity / norm, norm / dt

    def update(self):
        if self.is_matched:
            self.hits += 1
            self.time_since_update = 0
            self.kf.update()
            self.x, self.x_cov = self.get_state()

            if self.is_tentative() and self.hits >= self.init_age:
                self.track_state = TrackState.Confirmed

            elif self.is_rematched():
                self.track_state = TrackState.Confirmed

            elif self.is_lost():
                self.track_state = TrackState.Rematched

        else:
            self.hits = 0
            if self.is_tentative():
                self.track_state = TrackState.Ambiguous

            elif self.time_since_update >= self.max_age:
                self.track_state = TrackState.Ambiguous

            elif self.is_confirmed():
                self.track_state = TrackState.Lost

            if not self.is_valid_prediction():
                self.track_state = TrackState.Ambiguous

    def is_tentative(self):
        return self.track_state == TrackState.Tentative

    def is_confirmed(self):
        return self.track_state == TrackState.Confirmed or self.track_state == TrackState.Rematched

    def is_ambiguous(self):
        return self.track_state == TrackState.Ambiguous

    def is_rematched(self):
        return self.track_state == TrackState.Rematched

    def is_lost(self):
        return self.track_state == TrackState.Lost

    def is_valid_prediction(self):
        xyxy = self.state2xyxy()
        height = int(xyxy[3]) - int(xyxy[1])
        width = int(xyxy[2]) - int(xyxy[0])
        aspect_ratio = width / height
        area = width * height
        return aspect_ratio <= self.aspect_ratio_thr and area >= self.area_thr

    def state2xyxy(self, z=None):
        if z is None:
            projected_x = self.get_projected_state()[0]
        else:
            projected_x = z
        return self.measure2xyxy_func(projected_x)
