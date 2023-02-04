from typing import List
import numpy as np

from ..detection.base_detection import BaseDetection
from ..kalman_filter.config_kalman_filter import get_kalman_filter
from ..matching.config_matching import get_matching_fn
from ..track.config_track import get_track, BaseTrack
from ..cmc.base_cmc import BaseCMC


class BaseTracker:
    def __init__(self, trk_cfg):
        self.trk_cfg = trk_cfg
        self.matching_fn = get_matching_fn(trk_cfg)
        self.cmc = BaseCMC(
            method=trk_cfg.type_cmc,
            downscale=trk_cfg.cmc_downscale,
            use_cmc_file=trk_cfg.use_saved_cmc_result,
            cmc_result_dir=trk_cfg.cmc_results_dir
        )

        self.tracks: List[BaseTrack] = []
        self.track_id = 1

    def initialize(self):
        self.tracks = []
        self.track_id = 1

    def _init_tracks(self, detections: List[BaseDetection], unmatched_det_indices: List[int]):
        for det_idx in unmatched_det_indices:
            det = detections[det_idx]
            kf = get_kalman_filter(self.trk_cfg, det.z)
            trk = get_track(self.trk_cfg, self.track_id, det, kf)
            self.tracks.append(trk)
            self.track_id += 1

    def predict(self):
        for track in self.tracks:
            track.predict()

    def apply_cmc(self, img, vid_name: str = None, img_idx: int = None):
        warp = self.cmc.apply(img, vid_name, img_idx)
        R = warp[:2, :2]
        t = warp[:2, 2:3]
        for trk in self.tracks:
            tmp_cp = trk.x[:2]
            comp_cp = np.matmul(R, tmp_cp) + t
            trk.x[:2] = comp_cp
        return warp

    def update(self, detections: List[BaseDetection]):
        matched, unmatched_trk_indices, unmatched_det_indices = \
            self.match(detections)

        for det_idx, trk_idx in matched:
            tmp_trk = self.tracks[trk_idx]
            tmp_det = detections[det_idx]
            tmp_trk.measure(tmp_det, tmp_trk.is_lost() and self.trk_cfg.apply_oos)

        for track in self.tracks:
            track.update()

        self._init_tracks(detections, unmatched_det_indices)

        tracks = [track for track in self.tracks
                  if track.time_since_update <= track.max_age
                  and not (self.trk_cfg.delete_ambiguous and track.is_ambiguous())]
        self.tracks = tracks

    def match(self, detections: List[BaseDetection]):
        matched, unmatched_trk_indices, unmatched_det_indices = \
            self.matching_fn(
                trk_list=self.tracks,
                det_list=detections,
            )
        return matched, unmatched_trk_indices, unmatched_det_indices
