from typing import List
import numpy as np


class BaseDetection:
    def __init__(
            self,
            xyxy: List[float],  # [x1, y1, x2, y2]
            conf: float,
            cls: int,
            xyxy2measure_func,
            feature: np.ndarray = None,
    ):
        self.xyxy = np.asarray(xyxy, np.float32)
        self.conf = conf
        self.cls = cls
        self.feature = feature
        self.xyxy2measure_func = xyxy2measure_func

        self.is_matched = False
        self.z = self.xyxy2measure()

    def xyxy2measure(self):
        return self.xyxy2measure_func(self.xyxy)
