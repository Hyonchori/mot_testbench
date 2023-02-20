import os
import shutil
from pathlib import Path


class DetectorCFG:
    def __init__(self):
        self.tracker_name = 'test'

        # attributes for detector
        self.type_detector = 'yolox'
        self.use_detector = True
        self.use_saved_detector_result = False
        self.detector_weights = 'yolox_x_byte_ablation'
        self.detector_input_size = None
        self.detector_conf_thr = 0.01
        self.detector_iou_thr = 0.7

    @staticmethod
    def save_opt(save_dir):
        current_path = Path(__file__).absolute()
        save_path = os.path.join(save_dir, current_path.name)
        shutil.copyfile(current_path, save_path)
