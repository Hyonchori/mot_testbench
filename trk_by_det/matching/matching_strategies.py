import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List

from ..detection.base_detection import BaseDetection
from ..track.base_track import BaseTrack
from ..cost.costs import get_iou_cost, get_cosine_cost

from mot_testbench.custom_utils import plot_utils


def linear_assignment(cost_matrix, gate_matrix):
    tmp_rows, tmp_cols = linear_sum_assignment(cost_matrix)
    matched_rows = []
    matched_cols = []
    for row, col in zip(tmp_rows, tmp_cols):
        if gate_matrix[row, col] != 0:
            matched_rows.append(row)
            matched_cols.append(col)
    return matched_rows, matched_cols


def associate(
        cfg,
        trk_list: List[BaseTrack],
        det_list: List[BaseDetection],
        unmatched_trk_indices: List[int] = None,
        unmatched_det_indices: List[int] = None,
        img: np.ndarray = None  # for visualize cost and gate
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

    unmatched_trk_indices = list(set(unmatched_trk_indices) - set(matched_trk_indices))
    unmatched_det_indices = list(set(unmatched_det_indices) - set(matched_det_indices))

    matched_idx_pairs = [
        (det_idx, trk_idx) for det_idx, trk_idx in zip(matched_det_indices, matched_trk_indices)
    ]

    return matched_idx_pairs, unmatched_trk_indices, unmatched_det_indices


def associate_byte(
        cfg,
        trk_list: List[BaseTrack],
        det_list: List[BaseDetection],
        unmatched_trk_indices: List[int] = None,
        unmatched_det_indices: List[int] = None,
        img: np.ndarray = None  # for visualize cost and gate
):
    if unmatched_trk_indices is None:
        unmatched_trk_indices = list(range(len(trk_list)))
    if unmatched_det_indices is None:
        unmatched_det_indices = list(range(len(det_list)))

    if len(unmatched_trk_indices) == 0 or len(unmatched_det_indices) == 0:
        return [], unmatched_trk_indices, unmatched_det_indices

    high_conf_det_indices = [i for i, det in enumerate(det_list) if det.conf >= cfg.det_thr_high]
    low_conf_det_indices = [i for i, det in enumerate(det_list) if cfg.det_thr_low <= det.conf < cfg.det_thr_high]

    tracked_trk_indices = [i for i, trk in enumerate(trk_list) if trk.is_confirmed() or trk.is_lost()]
    untrakced_trk_indices = list(set(unmatched_trk_indices) - set(tracked_trk_indices))

    # first association
    cost_mat, gate_mat = get_iou_cost(trk_list, det_list, tracked_trk_indices, high_conf_det_indices)

    matched_det_indices1, matched_trk_indices1 = linear_assignment(cost_mat, gate_mat)
    matched_det_indices1 = [high_conf_det_indices[det_idx] for det_idx in matched_det_indices1]
    matched_trk_indices1 = [tracked_trk_indices[trk_idx] for trk_idx in matched_trk_indices1]

    unmatched_tracked_trk_indices = list(set(tracked_trk_indices) - set(matched_trk_indices1))
    unmatched_high_conf_det_indices = list(set(high_conf_det_indices) - set(matched_det_indices1))

    r_tracked_trk_indices = [i for i in unmatched_tracked_trk_indices if trk_list[i].is_confirmed()]

    # second association
    cost_mat, gate_mat = get_iou_cost(trk_list, det_list, r_tracked_trk_indices, low_conf_det_indices)
    matched_det_indices2, matched_trk_indices2 = linear_assignment(cost_mat, gate_mat)
    matched_det_indices2 = [low_conf_det_indices[det_idx] for det_idx in matched_det_indices2]
    matched_trk_indices2 = [r_tracked_trk_indices[trk_idx] for trk_idx in matched_trk_indices2]

    unmatched_r_tracked_trk_indices = list(set(r_tracked_trk_indices) - set(matched_trk_indices2))
    unmatched_low_conf_det_indices = list(set(low_conf_det_indices) - set(matched_det_indices2))

    # deal with unconfirmed tracks, usually tracks with only one beginning frame
    cost_mat, gate_mat = get_iou_cost(trk_list, det_list, untrakced_trk_indices, unmatched_high_conf_det_indices)
    matched_det_indices3, matched_trk_indices3 = linear_assignment(cost_mat, gate_mat)
    matched_det_indices3 = [unmatched_high_conf_det_indices[det_idx] for det_idx in matched_det_indices3]
    matched_trk_indices3 = [untrakced_trk_indices[trk_idx] for trk_idx in matched_trk_indices3]

    unmatched_trk_indices = list(set(unmatched_trk_indices) - set(matched_trk_indices1) - set(matched_trk_indices2) - set(matched_trk_indices3))
    unmatched_det_indices = list(set(unmatched_high_conf_det_indices) - set(matched_det_indices3))

    matched_det_indices = matched_det_indices1 + matched_det_indices2 + matched_det_indices3
    matched_trk_indices = matched_trk_indices1 + matched_trk_indices2 + matched_trk_indices3
    matched_idx_pairs = [
        (det_idx, trk_idx) for det_idx, trk_idx in zip(matched_det_indices, matched_trk_indices)
    ]

    return matched_idx_pairs, unmatched_trk_indices, unmatched_det_indices


def associate_bot(
        cfg,
        trk_list: List[BaseTrack],
        det_list: List[BaseDetection],
        unmatched_trk_indices: List[int] = None,
        unmatched_det_indices: List[int] = None,
        img: np.ndarray = None  # for visualize cost and gate
):
    if unmatched_trk_indices is None:
        unmatched_trk_indices = list(range(len(trk_list)))
    if unmatched_det_indices is None:
        unmatched_det_indices = list(range(len(det_list)))

    if len(unmatched_trk_indices) == 0 or len(unmatched_det_indices) == 0:
        return [], unmatched_trk_indices, unmatched_det_indices

    high_conf_det_indices = [i for i, det in enumerate(det_list) if det.conf >= cfg.det_thr_high]
    low_conf_det_indices = [i for i, det in enumerate(det_list) if cfg.det_thr_low <= det.conf < cfg.det_thr_high]

    tracked_trk_indices = [i for i, trk in enumerate(trk_list) if trk.is_confirmed() or trk.is_lost()]
    untrakced_trk_indices = list(set(unmatched_trk_indices) - set(tracked_trk_indices))

    # first association
    cost_mat_iou, gate_mat_iou = get_iou_cost(trk_list, det_list, tracked_trk_indices, high_conf_det_indices,
                                              cost_thr=0.5)
    cost_mat_cosine, gate_mat_cosine = get_cosine_cost(trk_list, det_list, tracked_trk_indices, high_conf_det_indices,
                                                       cost_thr=0.5)

    cost_mat_cosine /= 2.0
    cost_mat_cosine[gate_mat_cosine == 0] = 1.0
    cost_mat_cosine[gate_mat_iou == 0] = 1.0
    cost_mat = np.minimum(cost_mat_iou, cost_mat_cosine)

    gate_mat = np.zeros_like(gate_mat_iou)
    gate_mat[cost_mat < 0.8] = 1.0

    matched_det_indices1, matched_trk_indices1 = linear_assignment(cost_mat, gate_mat)
    matched_det_indices1 = [high_conf_det_indices[det_idx] for det_idx in matched_det_indices1]
    matched_trk_indices1 = [tracked_trk_indices[trk_idx] for trk_idx in matched_trk_indices1]

    unmatched_tracked_trk_indices = list(set(tracked_trk_indices) - set(matched_trk_indices1))
    unmatched_high_conf_det_indices = list(set(high_conf_det_indices) - set(matched_det_indices1))

    r_tracked_trk_indices = [i for i in unmatched_tracked_trk_indices if trk_list[i].is_confirmed()]

    # second association
    cost_mat, gate_mat = get_iou_cost(trk_list, det_list, r_tracked_trk_indices, low_conf_det_indices)
    matched_det_indices2, matched_trk_indices2 = linear_assignment(cost_mat, gate_mat)
    matched_det_indices2 = [low_conf_det_indices[det_idx] for det_idx in matched_det_indices2]
    matched_trk_indices2 = [r_tracked_trk_indices[trk_idx] for trk_idx in matched_trk_indices2]

    unmatched_r_tracked_trk_indices = list(set(r_tracked_trk_indices) - set(matched_trk_indices2))
    unmatched_low_conf_det_indices = list(set(low_conf_det_indices) - set(matched_det_indices2))

    # deal with unconfirmed tracks, usually tracks with only one beginning frame
    cost_mat, gate_mat = get_iou_cost(trk_list, det_list, untrakced_trk_indices, unmatched_high_conf_det_indices)
    matched_det_indices3, matched_trk_indices3 = linear_assignment(cost_mat, gate_mat)
    matched_det_indices3 = [unmatched_high_conf_det_indices[det_idx] for det_idx in matched_det_indices3]
    matched_trk_indices3 = [untrakced_trk_indices[trk_idx] for trk_idx in matched_trk_indices3]

    unmatched_trk_indices = list(
        set(unmatched_trk_indices) - set(matched_trk_indices1) - set(matched_trk_indices2) - set(matched_trk_indices3))
    unmatched_det_indices = list(set(unmatched_high_conf_det_indices) - set(matched_det_indices3))

    matched_det_indices = matched_det_indices1 + matched_det_indices2 + matched_det_indices3
    matched_trk_indices = matched_trk_indices1 + matched_trk_indices2 + matched_trk_indices3
    matched_idx_pairs = [
        (det_idx, trk_idx) for det_idx, trk_idx in zip(matched_det_indices, matched_trk_indices)
    ]

    return matched_idx_pairs, unmatched_trk_indices, unmatched_det_indices


def associate_test(
        cfg,
        trk_list: List[BaseTrack],
        det_list: List[BaseDetection],
        unmatched_trk_indices: List[int] = None,
        unmatched_det_indices: List[int] = None,
        img: np.ndarray = None  # for visualize cost and gate
):
    if unmatched_trk_indices is None:
        unmatched_trk_indices = list(range(len(trk_list)))
    if unmatched_det_indices is None:
        unmatched_det_indices = list(range(len(det_list)))

    if len(unmatched_trk_indices) == 0 or len(unmatched_det_indices) == 0:
        return [], unmatched_trk_indices, unmatched_det_indices

    high_conf_det_indices = [i for i, det in enumerate(det_list) if det.conf >= cfg.det_thr_high]
    low_conf_det_indices = [i for i, det in enumerate(det_list) if cfg.det_thr_low <= det.conf < cfg.det_thr_high]

    tracked_trk_indices = [i for i, trk in enumerate(trk_list) if trk.is_confirmed() or trk.is_lost()]
    untrakced_trk_indices = list(set(unmatched_trk_indices) - set(tracked_trk_indices))

    # first association
    cost_mat_iou, gate_mat_iou = get_iou_cost(trk_list, det_list, tracked_trk_indices, high_conf_det_indices)

    if cfg.use_extractor:
        cost_mat_cosine, gate_mat_cosine = get_cosine_cost(trk_list, det_list, tracked_trk_indices,
                                                           high_conf_det_indices,
                                                           cost_thr=0.5)
        # if img is not None:
        #     plot_utils.plot_cost(img,
        #                          [trk_list[i] for i in tracked_trk_indices],
        #                          [det_list[i] for i in high_conf_det_indices],
        #                          [cost_mat_cosine],
        #                          [gate_mat_cosine])
        iou_weight = np.ones_like(cost_mat_iou) * \
                     np.array([1.02 ** (trk_list[i].time_since_update - 1) for i in tracked_trk_indices])
        cost_mat = cost_mat_iou * iou_weight + cost_mat_cosine / 2.0
        gate_mat = gate_mat_iou + gate_mat_cosine
    else:
        cost_mat = cost_mat_iou
        gate_mat = gate_mat_iou

    matched_det_indices1, matched_trk_indices1 = linear_assignment(cost_mat, gate_mat)

    #matched_det_indices1, matched_trk_indices1 = linear_assignment(cost_mat, gate_mat)
    matched_det_indices1 = [high_conf_det_indices[det_idx] for det_idx in matched_det_indices1]
    matched_trk_indices1 = [tracked_trk_indices[trk_idx] for trk_idx in matched_trk_indices1]

    unmatched_tracked_trk_indices = list(set(tracked_trk_indices) - set(matched_trk_indices1))
    unmatched_high_conf_det_indices = list(set(high_conf_det_indices) - set(matched_det_indices1))

    r_tracked_trk_indices = [i for i in unmatched_tracked_trk_indices if trk_list[i].is_confirmed()]

    # second association
    cost_mat, gate_mat = get_iou_cost(trk_list, det_list, r_tracked_trk_indices, low_conf_det_indices)
    matched_det_indices2, matched_trk_indices2 = linear_assignment(cost_mat, gate_mat)
    matched_det_indices2 = [low_conf_det_indices[det_idx] for det_idx in matched_det_indices2]
    matched_trk_indices2 = [r_tracked_trk_indices[trk_idx] for trk_idx in matched_trk_indices2]

    unmatched_r_tracked_trk_indices = list(set(r_tracked_trk_indices) - set(matched_trk_indices2))
    unmatched_low_conf_det_indices = list(set(low_conf_det_indices) - set(matched_det_indices2))

    # deal with unconfirmed tracks, usually tracks with only one beginning frame
    cost_mat, gate_mat = get_iou_cost(trk_list, det_list, untrakced_trk_indices, unmatched_high_conf_det_indices)
    matched_det_indices3, matched_trk_indices3 = linear_assignment(cost_mat, gate_mat)
    matched_det_indices3 = [unmatched_high_conf_det_indices[det_idx] for det_idx in matched_det_indices3]
    matched_trk_indices3 = [untrakced_trk_indices[trk_idx] for trk_idx in matched_trk_indices3]

    unmatched_trk_indices = list(
        set(unmatched_trk_indices) - set(matched_trk_indices1) - set(matched_trk_indices2) - set(matched_trk_indices3))
    unmatched_det_indices = list(set(unmatched_high_conf_det_indices) - set(matched_det_indices3))

    matched_det_indices = matched_det_indices1 + matched_det_indices2 + matched_det_indices3
    matched_trk_indices = matched_trk_indices1 + matched_trk_indices2 + matched_trk_indices3
    matched_idx_pairs = [
        (det_idx, trk_idx) for det_idx, trk_idx in zip(matched_det_indices, matched_trk_indices)
    ]

    return matched_idx_pairs, unmatched_trk_indices, unmatched_det_indices


def associate_test2(
        cfg,
        trk_list: List[BaseTrack],
        det_list: List[BaseDetection],
        unmatched_trk_indices: List[int] = None,
        unmatched_det_indices: List[int] = None,
        img: np.ndarray = None  # for visualize cost and gate
):
    if unmatched_trk_indices is None:
        unmatched_trk_indices = list(range(len(trk_list)))
    if unmatched_det_indices is None:
        unmatched_det_indices = list(range(len(det_list)))

    if len(unmatched_trk_indices) == 0 or len(unmatched_det_indices) == 0:
        return [], unmatched_trk_indices, unmatched_det_indices

    high_conf_det_indices = [i for i, det in enumerate(det_list) if det.conf >= cfg.det_thr_high]
    low_conf_det_indices = [i for i, det in enumerate(det_list) if cfg.det_thr_low <= det.conf < cfg.det_thr_high]

    tracked_trk_indices = [i for i, trk in enumerate(trk_list) if trk.is_confirmed() or trk.is_lost()]
    untrakced_trk_indices = list(set(unmatched_trk_indices) - set(tracked_trk_indices))

    # first association
    cost_mat_iou, gate_mat_iou = get_iou_cost(trk_list, det_list, tracked_trk_indices, high_conf_det_indices)

    if cfg.use_extractor:
        cost_mat_cosine, gate_mat_cosine = get_cosine_cost(trk_list, det_list, tracked_trk_indices, high_conf_det_indices,
                                                           cost_thr=0.5)
        # if img is not None:
        #     plot_utils.plot_cost(img,
        #                          [trk_list[i] for i in tracked_trk_indices],
        #                          [det_list[i] for i in high_conf_det_indices],
        #                          [cost_mat_cosine],
        #                          [gate_mat_cosine])
        iou_weight = np.ones_like(cost_mat_iou) * \
                     np.array([1.02 ** (trk_list[i].time_since_update - 1) for i in tracked_trk_indices])
        cost_mat = cost_mat_iou * iou_weight + cost_mat_cosine / 2.0
        gate_mat = gate_mat_iou + gate_mat_cosine
    else:
        cost_mat = cost_mat_iou
        gate_mat = gate_mat_iou

    matched_det_indices1, matched_trk_indices1 = linear_assignment(cost_mat, gate_mat)

    #matched_det_indices1, matched_trk_indices1 = linear_assignment(cost_mat, gate_mat)
    matched_det_indices1 = [high_conf_det_indices[det_idx] for det_idx in matched_det_indices1]
    matched_trk_indices1 = [tracked_trk_indices[trk_idx] for trk_idx in matched_trk_indices1]

    unmatched_tracked_trk_indices = list(set(tracked_trk_indices) - set(matched_trk_indices1))
    unmatched_high_conf_det_indices = list(set(high_conf_det_indices) - set(matched_det_indices1))

    r_tracked_trk_indices = [i for i in unmatched_tracked_trk_indices if trk_list[i].is_confirmed()]

    # second association
    cost_mat, gate_mat = get_iou_cost(trk_list, det_list, r_tracked_trk_indices, low_conf_det_indices)
    matched_det_indices2, matched_trk_indices2 = linear_assignment(cost_mat, gate_mat)
    matched_det_indices2 = [low_conf_det_indices[det_idx] for det_idx in matched_det_indices2]
    matched_trk_indices2 = [r_tracked_trk_indices[trk_idx] for trk_idx in matched_trk_indices2]

    unmatched_r_tracked_trk_indices = list(set(r_tracked_trk_indices) - set(matched_trk_indices2))
    unmatched_low_conf_det_indices = list(set(low_conf_det_indices) - set(matched_det_indices2))

    # deal with unconfirmed tracks, usually tracks with only one beginning frame
    cost_mat, gate_mat = get_iou_cost(trk_list, det_list, untrakced_trk_indices, unmatched_high_conf_det_indices)
    matched_det_indices3, matched_trk_indices3 = linear_assignment(cost_mat, gate_mat)
    matched_det_indices3 = [unmatched_high_conf_det_indices[det_idx] for det_idx in matched_det_indices3]
    matched_trk_indices3 = [untrakced_trk_indices[trk_idx] for trk_idx in matched_trk_indices3]

    unmatched_trk_indices = list(
        set(unmatched_trk_indices) - set(matched_trk_indices1) - set(matched_trk_indices2) - set(matched_trk_indices3))
    unmatched_det_indices = list(set(unmatched_high_conf_det_indices) - set(matched_det_indices3))

    matched_det_indices = matched_det_indices1 + matched_det_indices2 + matched_det_indices3
    matched_trk_indices = matched_trk_indices1 + matched_trk_indices2 + matched_trk_indices3
    matched_idx_pairs = [
        (det_idx, trk_idx) for det_idx, trk_idx in zip(matched_det_indices, matched_trk_indices)
    ]

    return matched_idx_pairs, unmatched_trk_indices, unmatched_det_indices
