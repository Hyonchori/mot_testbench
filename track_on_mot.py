# Make tracking prediction on MOT17/MOT20 dataset using Tracking by Detection Algorithm

import argparse
import os
import sys
import time
import warnings
import importlib
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import tqdm

from custom_utils.general_utils import increment_path, xywh2xyxy
from custom_utils.torch_utils import select_device
# from custom_utils.plot_utils import letterbox, plot_info, plot_detection, plot_track

from datasets.MOT.get_mot_dataset import get_mot_videos, parsing_mot_detection, MOT_CLASSES

from trk_by_det.detector.config_detector import get_detector
from trk_by_det.detector.detector_utils import scale_coords, clip_coords
from trk_by_det.feature_extractor.config_extractor import get_extractor
from trk_by_det.detection.config_detection import init_detection_config, make_detection
from trk_by_det.kalman_filter.config_kalman_filter import init_kalman_filter_config
# from trk_by_det.tracker.base_tracker import BaseTracker


@torch.no_grad()
def main(args):
    # Arguments for MOT dataset
    mot_root = args.mot_root
    target_select = args.target_select
    target_split = args.target_split
    target_vid = args.target_vid
    target_det = args.target_det

    # Arguments for tracking by detection algorithm
    trk_cfg_file = args.trk_cfg_file
    sys.path.append((os.path.join(os.path.dirname(FILE), 'tracker_cfgs')))
    trk_cfg_file = importlib.import_module(trk_cfg_file)
    trk_cfg = trk_cfg_file.TrackerCFG()

    # General arguments for inference
    device = select_device(args.device)
    vis_progress_bar = args.vis_progress_bar
    out_dir = args.out_dir
    run_name = args.run_name
    vis_det = args.vis_det
    vis_trk = args.vis_trk
    vis_trk_debug = args.vis_trk_debug
    visualize = args.visualize
    view_size = args.view_size
    save_vid = args.save_vid
    save_pred = args.save_pred
    apply_only_matched = args.apply_only_matched

    # make save directory
    save_dir = increment_path(Path(out_dir) / run_name, exist_ok=False)
    if save_vid | save_pred:
        save_dir.mkdir(parents=True, exist_ok=True)
        trk_cfg.save_opt(save_dir)

    # load MOT dataset
    vid_root, vid_list, remain_dets = get_mot_videos(
        mot_root=mot_root,
        target_select=target_select,
        target_split=target_split,
        target_vid=target_vid,
        target_det=target_det
    )

    # load detector using config file name
    detector = get_detector(
        cfg=trk_cfg,
        device=device
    ) if trk_cfg.use_detector and not trk_cfg.use_saved_detector_result else None

    # load feature_extractor
    extractor = get_extractor(trk_cfg, device) if trk_cfg.type_extractor is not None else None

    # load tracker using config file name
    # tracker = BaseTracker(trk_cfg)
    init_detection_config(trk_cfg)
    init_kalman_filter_config(trk_cfg)
    # print(f"\nStart tracking using {trk_cfg.tracker_name}!")

    # iterate video
    for vid_idx, vid_name in enumerate(vid_list):
        print(f"\n--- Processing {vid_idx + 1} / {len(vid_list)}'s video: {vid_name}")
        if save_vid:
            vid_save_path = os.path.join(save_dir, f'{vid_name}_{trk_cfg.tracker_name}.mp4')
            if view_size is None:
                view_size = [720, 1280]
            vid_writer = cv2.VideoWriter(vid_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30,
                                         list(reversed(view_size)))
        else:
            vid_writer = None

        # init image iterator with detection result of MOT
        vid_dir = os.path.join(vid_root, vid_name)
        if trk_cfg.use_detector and trk_cfg.use_saved_detector_result:
            det_result_dir = f'{FILE.parents[0]}/trk_by_det/detector/{trk_cfg.type_detector}/results'
            if target_select == 'MOT17':
                det_file_name = '-'.join(vid_name.split('-')[:-1]) + '.txt'
            else:  # target_select == 'MOT20
                det_file_name = vid_name.split('.')[0] + '.txt'
            det_path = os.path.join(det_result_dir, f'{target_select}_{target_split}', trk_cfg.result_dir,
                                    det_file_name)
        else:
            det_path = os.path.join(vid_dir, 'det', 'det.txt')
        dets = parsing_mot_detection(det_path)

        img_dir = os.path.join(vid_dir, 'img1')
        img_names = sorted(os.listdir(img_dir))

        time.sleep(0.5)
        iterator = tqdm(enumerate(img_names), total=len(img_names), desc=vid_name) \
            if vis_progress_bar else enumerate(img_names)

        # initialize tracker when start each iteration
        # tracker.initialize()

        mot_trk_pred = ''
        for i, img_name in iterator:
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            img_v = img.copy()
            if not vis_progress_bar:
                print(f'\n--- {i + 1} / {len(img_names)}')

            # make detection (using detector or result from MOT17/MOT20)
            if trk_cfg.use_detector and not trk_cfg.use_saved_detector_result:
                det = detector(img)[0]
                if det is not None:
                    det = det.detach().cpu().numpy()
                    scale_coords(detector.input_size, det, img.shape)
                    if target_select == 'MOT20':
                        clip_coords(det, img.shape)
                else:
                    det = np.array([])
            else:
                det = np.asarray(dets[i])
                if len(det) != 0:
                    det = xywh2xyxy(det)

            detections = make_detection(trk_cfg, det, img, extractor)

            # show img_v
            if visualize:
                # plot_info(img_v, f'{vid_name}: {i + 1} / {len(img_names)}', font_size=1.2, font_thickness=2)
                cv2.imshow(vid_name, img_v)
                keyboard_input = cv2.waitKey(1) & 0xff
                if keyboard_input == ord('q'):
                    break
                elif keyboard_input == 27:  # 27: esc
                    sys.exit()

            # save img_v in video format
            if vid_writer is not None:
                vid_writer.write(img_v)

        if visualize:
            cv2.destroyWindow(vid_name)

        # save tracking prediction for TrackEval
        if save_pred:
            track_save_dir = os.path.join(save_dir, f'{target_select}-{target_split}', trk_cfg.tracker_name,
                                          'data')  # for using TrackEval code
            if not os.path.isdir(track_save_dir):
                os.makedirs(track_save_dir)

            pred_path = os.path.join(track_save_dir, f'{vid_name}.txt')
            with open(pred_path, 'w') as f:
                f.write(mot_trk_pred)
                # print(f'\ttrack prediction result is saved in "{pred_path}"!')

            if target_select == 'MOT17':
                for remain_det in remain_dets:
                    # for additional prediction file for TrackEval when using specific detection
                    remain_vid_name = '-'.join(vid_name.split('-')[:-1] + [remain_det])
                    remain_path = os.path.join(track_save_dir, f'{remain_vid_name}.txt')
                    with open(remain_path, 'w') as f:
                        f.write(mot_trk_pred)
                        # print(f'\ttrack prediction result is saved in "{remain_path}"!')
            time.sleep(0.05)



def get_args():
    parser = argparse.ArgumentParser()

    # Arguments for MOT17/MOT20 dataset
    mot_root = '/media/jhc/4AD250EDD250DEAF/dataset/mot'  # path to MOT dataset
    parser.add_argument('--mot_root', type=str, default=mot_root)

    target_select = 'MOT17'  # select in ['MOT17', 'MOT20']
    parser.add_argument('--target_select', type=str, default=target_select)

    target_split = 'test'  # select in ['train', 'test']
    parser.add_argument('--target_split', type=str, default=target_split)

    target_vid = None  # None: all videos, other numbers: target videos
    # for MOT17 train, select in [2, 4, 5, 9, 10, 11, 13]
    # for MOT17 test, select in [1, 3, 6, 7, 8, 12, 14]
    # for MOT20 train, select in [1, 2, 3, 5]
    # for MOT20 test, select in [4, 6, 7, 8]
    parser.add_argument('--target_vid', type=int, default=target_vid, nargs='+')

    target_det = ['FRCNN']  # for MOT17, select in ['DPM', 'FRCNN', 'SDP']
    parser.add_argument('--target_det', type=str, default=target_det, nargs='+')

    # Arguments for tracking by detection
    trk_cfg_file = 'test_trk'  # file name of target config in tracker_cfgs directory
    parser.add_argument('--trk_cfg_file', type=str, default=trk_cfg_file)

    # General arguments for inference
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--vis_progress_bar', action='store_true', default=True)
    parser.add_argument('--out_dir', type=str, default=f'{FILE.parents[0]}/runs/track_results/'
                                                       f'{target_select}_{target_split}')
    parser.add_argument('--run_name', type=str, default='SORT_Test')
    parser.add_argument('--vis_det', action='store_true', default=True)
    parser.add_argument('--vis_trk', action='store_true', default=True)
    parser.add_argument('--vis_trk_debug', action='store_true', default=False)
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--view_size', type=int, default=[720, 1280], nargs='+')  # [height, width]
    parser.add_argument('--save_vid', action='store_true', default=False)
    parser.add_argument('--save_pred', action='store_true', default=False)
    parser.add_argument('--apply_only_matched', action='store_true', default=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    FILE = Path(__file__).absolute()
    warnings.filterwarnings("ignore")
    np.set_printoptions(linewidth=np.inf)
    opt = get_args()
    main(opt)
