import os
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

from trk_by_det.detection.base_detection import BaseDetection
from trk_by_det.track.base_track import BaseTrack


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return[c[2], c[1], c[0]] if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return [int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)]


colors = Colors()  # create instance for 'from utils.plots import colors'


def plot_bboxes(
        img: np.ndarray,
        bboxes: np.ndarray,
        cls_names: dict = None,
        bbox_thickness: int = 2,
        hide_confidence: bool = False,
        font_size: float = 0.6,
        font_thickness: int = 2,
):
    for bbox in reversed(bboxes):  # reverse bboxes for sorting by confidence
        xyxy = list(map(int, bbox[:4]))
        conf = bbox[4]
        cls = int(bbox[5])
        color = [int(x * conf) for x in colors(cls, True)]
        cv2.rectangle(img, xyxy[:2], xyxy[2:], color, bbox_thickness)

        if cls_names is not None:
            label = f'{cls_names[cls]}'
        else:
            label = f'{cls}'
        label = label + f': {conf * 100:.1f}' if not hide_confidence else label
        plot_label(img, xyxy, label, color, font_size, font_thickness)


def plot_info(img, info, font_size=1, font_thickness=1):
    label_size = cv2.getTextSize(info, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)[0]
    cv2.rectangle(img, (0, 0), (label_size[0] + 10, label_size[1] * 2), [0, 0, 0], -1)
    cv2.putText(img, info, (5, int(label_size[1] * 1.5))
                , cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)


def plot_detection(
        img: np.ndarray,
        detections: List[BaseDetection],
        cls_names: dict = None,
        bbox_thickness: int = 2,
        hide_cls: bool = True,
        hide_confidence: bool = False,
        index_label: bool = False,
        hide_label: bool = False,
        font_size: float = 0.6,
        font_thickness: int = 2
):
    for i, det in enumerate(reversed(detections)):  # reverse bboxes for sorting by confidence
        xyxy = list(map(int, det.xyxy))
        conf = det.conf
        cls = det.cls
        color = [int(x * conf) for x in colors(cls, True)]
        cv2.rectangle(img, xyxy[:2], xyxy[2:], color, bbox_thickness)

        if hide_label:
            continue

        if cls_names is not None:
            label = f'{cls_names[cls]}' if not hide_cls else f'{cls}'
        else:
            label = f'{cls}'
        if index_label:
            label = f'{i}'
        label = label + f': {conf * 100:.1f}' if not hide_confidence else label
        plot_label(img, xyxy, label, color, font_size, font_thickness)


def plot_track(
        img: np.ndarray,
        tracks: List[BaseTrack],
        bbox_thickness: int = 2,
        center_size: int = 2,
        font_size: float = 0.6,
        font_thickness: int = 2,
        vis_vel: bool = True,
        vis_only_matched: bool = False,
        empty_name: bool = False,
):
    ref_img = np.zeros_like(img)
    for trk in tracks:
        xyxy = list(map(int, trk.state2xyxy()))
        track_id = trk.track_id
        color = colors(track_id, True)
        is_matched = trk.is_matched

        if is_matched:
            cv2.rectangle(img, xyxy[:2], xyxy[2:], color, bbox_thickness)
            text = f'{track_id}' if not empty_name else ''
            plot_label(img, xyxy, text, color, font_size, font_thickness)
        else:
            if not vis_only_matched:
                cv2.rectangle(ref_img, xyxy[:2], xyxy[2:], color, -1)
                text = f'{track_id}: {trk.time_since_update}'
                plot_label(ref_img, xyxy, text, color, font_size, font_thickness)
            else:
                continue

        trk_z, trk_z_cov = trk.get_projected_state()
        cx, cy = trk_z[:2, 0]
        if vis_vel:
            direction = trk.velocity
        else:
            direction = trk.direction
        plot_arrow(img, np.array([cx, cy]), direction, color, vis_vel)
        plot_center(img, xyxy, color, center_size)

    add_img = cv2.addWeighted(img, 1., ref_img, 0.5, 0.1)
    return add_img


def plot_arrow(
        img: np.ndarray,
        start: np.ndarray,
        direction: np.ndarray,
        color: List[int] = (255, 255, 255),
        is_vel: bool = True,
        line_thickness: int = 2,
        arrow_size: int = 10
):
    if is_vel:
        end = start + direction
        direction = direction / (np.linalg.norm(direction) + 1e-6)
    else:
        end = start + direction * arrow_size
    cv2.line(img,
             (int(start[0]), int(start[1])),
             (int(end[0]), int(end[1])),
             color, line_thickness)
    left_rot_mat = get_rotation_matrix(- np.pi / 4)
    right_rot_mat = get_rotation_matrix(np.pi / 4)
    left_direction = np.matmul(left_rot_mat, - direction)
    right_direction = np.matmul(right_rot_mat, - direction)
    left_end = end + left_direction * arrow_size * 0.5
    right_end = end + right_direction * arrow_size * 0.5
    cv2.line(img,
             (int(end[0]), int(end[1])),
             (int(left_end[0]), int(left_end[1])),
             color, line_thickness)
    cv2.line(img,
             (int(end[0]), int(end[1])),
             (int(right_end[0]), int(right_end[1])),
             color, line_thickness)


def get_rotation_matrix(radians: float):
    rotation_matrix = np.array([
        [np.cos(radians), - np.sin(radians)],
        [np.sin(radians), np.cos(radians)]
    ])
    return rotation_matrix


def plot_center(
        img: np.ndarray,
        xyxy: List[int],
        color: List[int],
        center_size: int = 2
):
    center = [int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)]
    cv2.circle(img, center, center_size, color, -1)


def plot_label(
        img: np.ndarray,
        xyxy: List[int],
        label: str,
        color: List[int],
        font_size: float = 0.6,
        font_thickness: int = 1
):
    if not isinstance(label, str):
        label = str(label)

    txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)[0]
    txt_bk_color = [int(c * 0.7) for c in color]
    cv2.rectangle(img, xyxy[:2], (xyxy[0] + txt_size[0] + 1, xyxy[1] - int(txt_size[1] * 1.5)),
                  txt_bk_color, -1)
    cv2.putText(img, label, (xyxy[0], xyxy[1] - int(txt_size[1] * 0.4)),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness)


def letterbox(
        img: np.ndarray,
        new_shape=(640, 640),
        color: int = (114, 114, 114),
        auto: bool = True,
        stretch: bool = False,
        stride: int = 32,
        dnn_pad: bool = False
):
    # resize and pad image while meeting stride-multiple constraints
    shape = img.shape[: 2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    if dnn_pad:
        new_shape = [x + (x + stride) % stride for x in new_shape]

    if img.shape[:2] == new_shape:
        return img, 1., (0, 0)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif stretch:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return img, ratio, (dw, dh)

