import os
import shutil
from pathlib import Path


class TrackerCFG:
    def __init__(self):
        self.tracker_name = 'test'

        # attributes for detector
        self.type_detector = 'yolox'
        self.use_detector = True
        self.use_saved_detector_result = True
        self.result_dir = 'yolox_x_byte_mot17'
        self.detector_weights = 'yolox_x_byte_mot17'
        self.detector_input_size = None
        self.detector_conf_thr = 0.01
        self.detector_iou_thr = 0.7

        # attributes for feature extractor
        self.type_extractor = 'fast_reid'
        self.use_extractor = True
        self.extractor_input_size = None

        # attributes for CMC
        self.type_cmc = 'sparseOptFlow'  # select in [None, 'ecc', 'sparseOptFlow']
        self.use_cmc = True
        self.cmc_downscale = 2.0
        self.use_saved_cmc_result = True
        self.cmc_results_dir = '/home/jhc/PycharmProjects/pythonProject/SORT_FAMILY/BoT-SORT/tracker/GMC_files/MOTChallenge'

        # attributes for track
        self.det_thr_low = 0.1
        self.det_thr_high = 0.6
        self.aspect_ratio_thr = 1.6
        self.area_thr = 100
        self.type_state = 'cpwh'
        self.type_kalman_filter = 'deep_sort'
        self.type_feature = None
        self.max_age = 30
        self.init_age = 3
        self.feature_gallery_len = 1
        self.ema_alpha = 0.9
        self.time_difference = 3
        self.apply_oos = False
        self.type_matching = 'test'
        self.std_weight_position = 1. / 20
        self.std_weight_velocity = 1. / 160
        self.is_nsa = False
        self.delete_ambiguous = True
        self.apply_obs_to_lost = False
        self.confirm_by_conf = False

    @staticmethod
    def save_opt(save_dir):
        current_path = Path(__file__).absolute()
        save_path = os.path.join(save_dir, current_path.name)
        shutil.copyfile(current_path, save_path)
