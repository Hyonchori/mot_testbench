# Make detection prediction on MOT17/MOT20 dataset for using in Tracking by Detection Algorithm

import argparse
import importlib
import os
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from custom_utils.general_utils import increment_path, xyxy2xywh
from custom_utils.torch_utils import select_device
from custom_utils.plot_utils import letterbox, plot_bboxes, plot_info

from datasets.MOT.get_mot_dataset import get_mot_videos, MOT_CLASSES

from trk_by_det.detector.config_detector import get_detector
from trk_by_det.detector.detector_utils import scale_coords, clip_coords


def main(args):
    # Arguments for MOT dataset
    mot_root = args.mot_root
    target_select = args.target_select
    target_split = args.target_split
    target_vid = args.target_vid

    # Arguments for detector
    det_cfg_file = args.det_cfg_file
    sys.path.append((os.path.join(os.path.dirname(FILE), 'detector_cfgs')))
    det_cfg_file = importlib.import_module(det_cfg_file)
    det_cfg = det_cfg_file.DetectorCFG()

    # General arguments for inference
    device = select_device(args.device)
    vis_progress_bar = args.vis_progress_bar
    run_name = args.run_name
    visualize = args.visualize
    view_size = args.view_size
    save_vid = args.save_vid
    save_pred = args.save_pred

    # make save directory
    out_dir = f'{FILE.parents[0]}/trk_by_det/detector/{det_cfg.type_detector}/results/{target_select}_{target_split}'
    save_dir = increment_path(Path(out_dir) / run_name, exist_ok=False)
    if save_vid | save_pred:
        save_dir.mkdir(parents=True, exist_ok=True)
        det_cfg.save_opt(save_dir)

    # load MOT dataset
    vid_root, vid_list, remain_dets = get_mot_videos(
        mot_root=mot_root,
        target_select=target_select,
        target_split=target_split,
        target_vid=target_vid,
        target_det=['FRCNN']
    )

    # load detector using config file name
    detector = get_detector(
        cfg=det_cfg,
        device=device
    )

    # iterate video
    for vid_idx, vid_name in enumerate(vid_list):
        print(f"\n--- Processing {vid_idx + 1} / {len(vid_list)}'s video: {vid_name}")
        if save_vid:
            vid_save_path = os.path.join(save_dir, f'{vid_name}_{det_cfg.detector_name}.mp4')
            if view_size is None:
                view_size = [720, 1280]
            vid_writer = cv2.VideoWriter(vid_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30,
                                         list(reversed(view_size)))
        else:
            vid_writer = None

        # init image iterator
        vid_dir = os.path.join(vid_root, vid_name)
        img_dir = os.path.join(vid_dir, 'img1')
        img_names = sorted(os.listdir(img_dir))

        time.sleep(0.5)
        iterator = tqdm(enumerate(img_names), total=len(img_names), desc=vid_name) \
            if vis_progress_bar else enumerate(img_names)

        mot_det_pred = ''
        for i, img_name in iterator:
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            img_v = img.copy()
            if not vis_progress_bar:
                print('\n')

            # make detection using detector
            det = detector(img)[0]
            if det is not None:
                det = det.detach().cpu().numpy()
                scale_coords(detector.input_size, det, img.shape)
                if target_select == 'MOT20':
                    clip_coords(det, img.shape)

                bboxes = xyxy2xywh(det)
                for bbox in bboxes:
                    mot_det_pred += f'{i + 1},-1,{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{bbox[4]}\n'
            else:
                det = []

            if visualize:
                plot_info(img_v, f'{vid_name}: {i + 1} / {len(img_names)}', font_size=1.2, font_thickness=2)
                plot_bboxes(img_v, det, MOT_CLASSES)

                if view_size is not None:
                    img_v = letterbox(img_v, view_size)[0]

                cv2.imshow(vid_name, img_v)
                keyboard_input = cv2.waitKey(1) & 0xff
                if keyboard_input == ord('q'):
                    break
                elif keyboard_input == 27:  # 27: esc
                    sys.exit()

            if vid_writer is not None:
                vid_writer.write(img_v)

        if visualize:
            cv2.destroyWindow(vid_name)

        if save_pred:
            if target_select == 'MOT17':
                pred_save_path = os.path.join(save_dir, '-'.join(vid_name.split('-')[:-1]) + '.txt')
            else:  # target_select == 'MOT20'
                pred_save_path = os.path.join(save_dir, vid_name.split('.')[0] + '.txt')
            with open(pred_save_path, 'w') as f:
                f.write(mot_det_pred)


def get_args():
    parser = argparse.ArgumentParser()

    # Arguments for MOT dataset
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

    # Arguments for detector
    det_cfg_file = 'test_det'  # file name of target config in detector_cfgs directory
    parser.add_argument('--det_cfg_file', type=str, default=det_cfg_file)

    # General arguments for inference
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--vis_progress_bar', action='store_true', default=True)
    parser.add_argument('--run_name', type=str, default='yolox_x_byte_mot17')
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--view_size', type=int, default=[720, 1280], nargs='+')  # [height, width]
    parser.add_argument('--save_vid', action='store_true', default=False)
    parser.add_argument('--save_pred', action='store_true', default=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    FILE = Path(__file__).absolute()
    warnings.filterwarnings("ignore")
    np.set_printoptions(linewidth=np.inf)
    opt = get_args()
    main(opt)
