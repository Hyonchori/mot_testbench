import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List

from ..detection.base_detection import BaseDetection
from ..track.base_track import BaseTrack
from ..cost.costs import get_iou_cost, get_cosine_cost


def linear_assignment(cost_matrix, gate_matrix):
    tmp_rows, tmp_cols = linear_sum_assignment(cost_matrix)
    matched_rows = []
    matched_cols = []
    for row, col in zip(tmp_rows, tmp_cols):
        if gate_matrix[row, col] == 1:
            matched_rows.append(row)
            matched_cols.append(col)
    return matched_rows, matched_cols


def associate(
        trk_list: List[BaseTrack],
        det_list: List[BaseDetection],
        unmatched_trk_indices: List[int] = None,
        unmatched_det_indices: List[int] = None
):
    if unmatched_trk_indices is None:
        unmatched_trk_indices = list(range(len(trk_list)))
    if unmatched_det_indices is None:
        unmatched_det_indices = list(range(len(det_list)))

    if len(unmatched_trk_indices) == 0 or len(unmatched_det_indices) == 0:
        return [], unmatched_trk_indices, unmatched_det_indices

    cost_mat, gate_mat = get_iou_cost(trk_list, det_list, unmatched_trk_indices, unmatched_det_indices)

    matched_det_indices, matched_trk_indices = linear_assignment(cost_mat, gate_mat)

    matched_det_indices = [unmatched_det_indices[det_idx] for det_idx in matched_det_indices]
    matched_trk_indices = [unmatched_trk_indices[trk_idx] for trk_idx in matched_trk_indices]

    matched_idx_pairs = [
        (det_idx, trk_idx) for det_idx, trk_idx in zip(matched_det_indices, matched_trk_indices)
    ]
    unmatched_trk_indices = list(set(unmatched_trk_indices) - set(matched_trk_indices))
    unmatched_det_indices = list(set(unmatched_det_indices) - set(matched_det_indices))

    return matched_idx_pairs, unmatched_trk_indices, unmatched_det_indices

