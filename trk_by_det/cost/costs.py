from typing import List
import numpy as np

from ..track.base_track import BaseTrack
from ..detection.base_detection import BaseDetection


def get_iou_distance(
        trk_xyxy: np.ndarray,  # (4)
        det_xyxy: np.ndarray,  # (4)
):
    overlap_x = min(trk_xyxy[2], det_xyxy[2]) - max(trk_xyxy[0], det_xyxy[0])
    overlap_y = min(trk_xyxy[3], det_xyxy[3]) - max(trk_xyxy[1], det_xyxy[1])

    if overlap_x > 0.0 and overlap_y > 0.0:
        inter_area = overlap_x * overlap_y
        trk_area = (trk_xyxy[2] - trk_xyxy[0]) * (trk_xyxy[3] - trk_xyxy[1])
        det_area = (det_xyxy[2] - det_xyxy[0]) * (det_xyxy[3] - det_xyxy[1])
        iou = inter_area / (trk_area + det_area - inter_area)
    else:
        iou = 0.0
    return 1.0 - iou


def get_iou_cost(
        tracks: List[BaseTrack],
        detections: List[BaseDetection],
        trk_indices: List[int] = None,
        det_indices: List[int] = None,
        cost_thr: float = 0.66
):
    if trk_indices is None:
        trk_indices = np.arange(len(tracks))
    if det_indices is None:
        det_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(det_indices), len(trk_indices)), dtype=np.float32)
    gate_matrix = np.zeros_like(cost_matrix)

    for row, det_idx in enumerate(det_indices):
        det_xyxy = detections[det_idx].xyxy
        for col, trk_idx in enumerate(trk_indices):
            trk_xyxy = tracks[trk_idx].state2xyxy()
            iou_distance = get_iou_distance(trk_xyxy, det_xyxy)
            cost_matrix[row, col] = iou_distance
            if iou_distance < cost_thr:
                gate_matrix[row, col] = 1.0

    return cost_matrix, gate_matrix


def get_cosine_distance(
        trk_features: np.ndarray,  # N x M matrix of N samples of dimensionality M
        det_features: np.ndarray,  # L x M matrix of L samples of dimensionality M
        is_normalized: bool = True
):
    if not is_normalized:
        trk_features = np.asarray(trk_features) / np.linalg.norm(trk_features, axis=1, keepdims=True)
        det_features = np.asarray(det_features) / np.linalg.norm(det_features, axis=1, keepdims=True)
    distances = 1. - np.dot(trk_features, det_features.T)
    return distances.min(axis=0)


def get_cosine_cost(
        tracks: List[BaseTrack],
        detections: List[BaseDetection],
        trk_indices: List[int] = None,
        det_indices: List[int] = None,
        cost_thr: float = 0.2
):
    cost_matrix = np.zeros((len(det_indices), len(trk_indices)), dtype=np.float32)
    gate_matrix = np.zeros_like(cost_matrix)

    for row, det_idx in enumerate(det_indices):
        det_feat = detections[det_idx].feature[None]
        for col, trk_idx in enumerate(trk_indices):
            trk_feat = np.asarray(tracks[trk_idx].features)
            cosine_distance = get_cosine_distance(trk_feat, det_feat)
            cost_matrix[row, col] = cosine_distance
            if cosine_distance < cost_thr:
                gate_matrix[row, col] = 1.0

    return cost_matrix, gate_matrix



