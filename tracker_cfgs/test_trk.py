import os
import shutil
from pathlib import Path


class TrackerCFG:
    def __init__(self):
        self.tracker_name = 'test'

        # attributes for detector
        self.type_detector = 'yolox'
        self.use_detector = True
        self.use_saved_detector_result = False
        self.detector_weights = 'yolox_x_byte_mot17'
        self.detector_input_size = None
        self.detector_conf_thr = 0.7
        self.detector_iou_thr = 0.01

        # attributes for feature extractor
        self.type_extractor = 'simple_cnn'
        self.extractor_input_size = None

        # attributes for track
        self.det_thr_low = 0.1
        self.det_thr_high = 0.7
        self.aspect_ratio_thr = 1.6
        self.area_thr = 100
        self.type_state = 'cpah'
        self.type_kalman_filter = 'deep_sort'

    @staticmethod
    def save_opt(save_dir):
        current_path = Path(__file__).absolute()
        save_path = os.path.join(save_dir, current_path.name)
        shutil.copyfile(current_path, save_path)
