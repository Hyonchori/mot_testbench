import torch
import numpy as np
from typing import List

from .base_detection import BaseDetection


detection_config = {}


def init_detection_config(cfg):
    if cfg.type_state == 'cpsa':
        def xyxy2measure(xyxy):
            width = max(1., xyxy[2] - xyxy[0])
            height = max(1., xyxy[3] - xyxy[1])
            cpsa = [
                [xyxy[0] + 0.5 * width],
                [xyxy[1] + 0.5 * height],
                [width * height],
                [width / height]
            ]
            return np.asarray(cpsa)  # [center_x, center_y, area, aspect ratio]

    elif cfg.type_state == 'cpah':
        def xyxy2measure(xyxy):
            width = max(1., xyxy[2] - xyxy[0])
            height = max(1., xyxy[3] - xyxy[1])
            cpah = [
                [xyxy[0] + 0.5 * width],
                [xyxy[1] + 0.5 * height],
                [width / height],
                [height]
            ]
            return np.asarray(cpah)  # [center_x, center_y, aspect ratio, height]

    elif cfg.type_state == 'cpwh':
        def xyxy2measure(xyxy):
            width = max(1., xyxy[2] - xyxy[0])
            height = max(1., xyxy[3] - xyxy[1])
            cpwh = [
                [xyxy[0] + 0.5 * width],
                [xyxy[1] + 0.5 * height],
                [width],
                [height]
            ]
            return np.asarray(cpwh)  # [center_x, center_y, width, height]

    else:
        def xyxy2measure(xyxy):
            width = max(1., xyxy[2] - xyxy[0])
            height = max(1., xyxy[3] - xyxy[1])
            cpsa = [
                [xyxy[0] + 0.5 * width],
                [xyxy[1] + 0.5 * height],
                [width * height],
                [width, height]
            ]
            return np.asarray(cpsa, dtype=np.float32)  # [center_x, center_y, area, aspect ratio]

    detection_config['xyxy2measure_func'] = xyxy2measure


def get_detection(
            xyxy: List[float],  # [x1, y1, x2, y2]
            conf: float,
            cls: int,
            feature: np.ndarray = None,
):
    return BaseDetection(
        xyxy=xyxy,
        conf=conf,
        cls=cls,
        xyxy2measure_func=detection_config['xyxy2measure_func'],
        feature=feature
    )


def is_valid_detection(xyxy, conf, det_thr, aspect_ratio_thr, area_thr):
    # height = int(xyxy[3]) - int(xyxy[1])
    # width = int(xyxy[2]) - int(xyxy[0])
    # aspect_ratio = width / height
    # area = width * height
    return conf >= det_thr  # and aspect_ratio <= aspect_ratio_thr and area >= area_thr


@torch.no_grad()
def make_detection(
        cfg,
        predictions: np.ndarray,  # [x1, y1, x2, y2, confidence, class]
        img: np.ndarray = None,  # (height, width, channels)
        extractor=None
) -> List[BaseDetection]:
    detections = []

    if extractor is None:
        for res in predictions:
            bbox = res[:4]
            conf = float(res[4])
            cls = int(res[5])
            if is_valid_detection(bbox, conf, cfg.det_thr_low, cfg.aspect_ratio_thr, cfg.area_thr):
                tmp_det = get_detection(xyxy=bbox, conf=conf, cls=cls)
                detections.append(tmp_det)
    else:
        img_crops = []
        valid_predictions = []
        for i, res in enumerate(predictions):
            bbox = res[:4]
            conf = float(res[4])
            if is_valid_detection(bbox, conf, cfg.det_thr_low, cfg.aspect_ratio_thr, cfg.area_thr):
                img_crop = img[max(0, int(bbox[1])): int(bbox[3]), max(0, int(bbox[0])): int(bbox[2])]
                valid_predictions.append(i)
                img_crops.append(img_crop)

        if len(img_crops) > 0:
            features = extractor(img_crops)
            for res_idx, feat in zip(valid_predictions, features):
                res = predictions[res_idx]
                bbox = res[:4]
                conf = float(res[4])
                cls = int(res[5])
                tmp_det = get_detection(xyxy=bbox, conf=conf, cls=cls, feature=feat)
                detections.append(tmp_det)

    return detections
