# Make tracking prediction on MOT17/MOT20 dataset using reference tracking algorithms

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

from custom_utils.general_utils import increment_path, xywh2xyxy
from custom_utils.torch_utils import select_device
from custom_utils.plot_utils import letterbox, plot_info, plot_bboxes

from datasets.MOT.get_mot_dataset import get_mot_videos, parsing_mot_detection

from trk_by_det.detector.config_detector import get_detector
from trk_by_det.detector.detector_utils import scale_coords, clip_coords


REFERENCE_TRACKERS = {
    1: 'BYTE',
    2: 'OC-SORT',
    3: 'SORT',
    4: 'DeepSORT',
    5: 'BotSORT'
}


def main(args):
    # Arguments for MOT dataset
    mot_root = args.mot_root
    target_select = args.target_select
    target_split = args.target_split
    target_vid = args.target_vid
    target_det = args.target_det

    # Arguments for detector
    det_cfg_file = args.det_cfg_file
    use_detector = args.use_detector
    use_saved_detector_result = args.use_saved_detector_result
    detector_result_dir = args.detector_result_dir
    sys.path.append((os.path.join(os.path.dirname(FILE), 'detector_cfgs')))
    det_cfg_file = importlib.import_module(det_cfg_file)
    det_cfg = det_cfg_file.DetectorCFG()

    # Arguments for reference tracker
    select_tracker = args.select_tracker

    # General arguments for inference
    device = select_device(args.device)
    vis_progress_bar = args.vis_progress_bar
    out_dir = args.out_dir
    run_name = args.run_name
    vis_det = args.vis_det
    vis_trk = args.vis_trk
    visualize = args.visualize
    view_size = args.view_size
    save_vid = args.save_vid
    save_pred = args.save_pred

    # make save directory
    save_dir = increment_path(Path(out_dir) / run_name, exist_ok=False)
    if save_vid | save_pred:
        save_dir.mkdir(parents=True, exist_ok=True)

    # load MOT dataset
    vid_root, vid_list, remain_dets = get_mot_videos(
        mot_root=mot_root,
        target_select=target_select,
        target_split=target_split,
        target_vid=target_vid,
        target_det=target_det
    )

    # load detector using config file name
    detector = get_detector(type_detector=det_cfg.type_detector)(
        det_cfg=det_cfg,
        device=device
    ) if use_detector and not use_saved_detector_result else None

    # load tracker using config file name
    if select_tracker == 1:  # load BYTE tracker
        print('\nBYTE trakcer is selected!')
        from ref_trackers.byte_tracker.byte_args import make_byte_args
        from ref_trackers.byte_tracker.byte_tracker import BYTETracker
        tracker_args = make_byte_args()
        tracker = BYTETracker(tracker_args)

    elif select_tracker == 2:  # load OC-SORT tracker
        print('\nOC-SORT trakcer is selected!')
        from ref_trackers.ocsort_tracker.oc_args import make_oc_args
        from ref_trackers.ocsort_tracker.ocsort import OCSort
        tracker_args = make_oc_args()
        tracker = OCSort(
            det_thresh=tracker_args.track_thresh,
            iou_threshold=tracker_args.iou_thresh,
            asso_func=tracker_args.asso,
            delta_t=tracker_args.deltat,
            inertia=tracker_args.inertia
        )

    elif select_tracker == 3:  # load SORT tracker
        print('\nSORT trakcer is selected!')
        from ref_trackers.sort_tracker.sort_args import make_sort_args
        from ref_trackers.sort_tracker.sort import Sort
        tracker_args = make_sort_args()
        tracker = Sort(
            det_thresh=tracker_args.track_thresh
        )

    elif select_tracker == 4:  # load DeepSORT tracker
        print('\nDeepSORT trakcer is selected!')
        from ref_trackers.deepsort_tracker.deepsort_args import make_deepsort_args
        from ref_trackers.deepsort_tracker.deepsort import DeepSort
        tracker_args = make_deepsort_args()
        tracker = DeepSort(
            model_path=tracker_args.model_path,
            min_confidence=tracker_args.track_thresh
        )

    elif select_tracker == 5:  # load BOT-SORT tracker
        print('\nBotSORT trakcer is selected!')
        from ref_trackers.botsort_tracker.bot_args import make_bot_args
        from ref_trackers.botsort_tracker.bot_sort import BoTSORT
        tracker_args = make_bot_args()
        tracker = BoTSORT(
            tracker_args
        )

    # iterate video
    start_time = time.time()
    for vid_idx, vid_name in enumerate(vid_list):
        print(f"\n--- Processing {vid_idx + 1} / {len(vid_list)}'s video: {vid_name}")
        if save_vid:
            vid_save_path = os.path.join(save_dir, f'{vid_name}_{REFERENCE_TRACKERS[select_tracker]}.mp4')
            if view_size is None:
                view_size = [720, 1280]
            vid_writer = cv2.VideoWriter(vid_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30,
                                         list(reversed(view_size)))
        else:
            vid_writer = None

        # init image iterator with detection result of MOT
        vid_dir = os.path.join(vid_root, vid_name)
        if use_detector and use_saved_detector_result:
            det_result_dir = f'{FILE.parents[0]}/trk_by_det/detector/{det_cfg.type_detector}/results'
            if target_select == 'MOT17':
                det_file_name = '-'.join(vid_name.split('-')[:-1]) + '.txt'
            else:  # target_select == 'MOT20
                det_file_name = vid_name.split('.')[0] + '.txt'
            det_path = os.path.join(det_result_dir, f'{target_select}_{target_split}', detector_result_dir,
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
        if select_tracker == 1:
            if vid_name == 'MOT17-05-FRCNN' or vid_name == 'MOT17-06-FRCNN':
                tracker_args.track_buffer = 14
            elif vid_name == 'MOT17-13-FRCNN' or vid_name == 'MOT17-14-FRCNN':
                tracker_args.track_buffer = 25
            else:
                tracker_args.track_buffer = 30

            if vid_name == 'MOT17-01-FRCNN':
                tracker_args.track_thresh = 0.65
            elif vid_name == 'MOT17-06-FRCNN':
                tracker_args.track_thresh = 0.65
            elif vid_name == 'MOT17-12-FRCNN':
                tracker_args.track_thresh = 0.7
            elif vid_name == 'MOT17-14-FRCNN':
                tracker_args.track_thresh = 0.67
            elif vid_name in ['MOT20-06', 'MOT20-08']:
                tracker_args.track_thresh = 0.3
            else:
                tracker_args.track_thresh = 0.6
            tracker.initialize(tracker_args)
        elif select_tracker == 5:
            tracker.initialize(tracker_args)
        else:
            tracker.initialize()

        mot_trk_pred = ''
        for i, img_name in iterator:
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            img_v = img.copy()
            if not vis_progress_bar:
                print('\n')

            # make detection (using detector or result from MOT17/MOT20)
            if use_detector and not use_saved_detector_result:
                det = detector(img)[0]
                if det is not None:
                    det = det.detach().cpu().numpy()
                    scale_coords(detector.input_size, det, img.shape)
                    if target_select == 'MOT20':
                        clip_coords(det, img.shape)
                else:
                    det = np.empty((0, 6))
            else:
                det = np.asarray(dets[i])
                if len(det) != 0:
                    det = xywh2xyxy(det)
                else:
                    det = np.empty((0, 6))

            if vis_det:
                plot_bboxes(img_v, det, hide_confidence=True)

            pred = []
            # update BYTE
            if select_tracker == 1:
                online_targets = tracker.update(det, img.shape[:2], img.shape[:2])
                for t in online_targets:
                    tlwh = t.tlwh
                    conf = t.score
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > tracker_args.vertical_thresh
                    if tlwh[2] * tlwh[3] > tracker_args.min_box_area and not vertical:
                        pred.append([*t.tlbr, conf, tid])

            # update OC-SORT / SORT
            elif select_tracker == 2 or select_tracker == 3:
                online_targets = tracker.update(det, img.shape[:2], img.shape[:2])
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2]-t[0], t[3]-t[1]]
                    tid = t[4]
                    vertical = tlwh[2] / tlwh[3] > tracker_args.vertical_thresh
                    if tlwh[2] * tlwh[3] > tracker_args.min_box_area and not vertical:
                        pred.append([*t[:4], 1, tid])

            # update DeepSORT
            elif select_tracker == 4:
                online_targets = tracker.update(det, img.shape[:2], img.shape[:2], img)
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2]-t[0], t[3]-t[1]]
                    tid = t[4]
                    vertical = tlwh[2] / tlwh[3] > tracker_args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > tracker_args.min_box_area and not vertical:
                        pred.append([*t[:4], 1, tid])
                        
            # update BoTSORT
            elif select_tracker == 5:
                online_targets = tracker.update(det, img)
                for t in online_targets:
                    tlwh = t.tlwh
                    conf = t.score
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > tracker_args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > tracker_args.min_box_area and not vertical:
                        pred.append([*t.tlbr, conf, tid])

            if vis_trk:
                plot_bboxes(img_v, pred, hide_confidence=True)

            if save_pred:
                for p in pred:
                    trk_xyxy = p[:4]
                    trk_id = p[5]
                    trk_width = trk_xyxy[2] - trk_xyxy[0]
                    trk_height = trk_xyxy[3] - trk_xyxy[1]
                    mot_trk_pred += f'{i + 1},{trk_id},' + \
                        f'{trk_xyxy[0]},{trk_xyxy[1]},{trk_width},{trk_height},-1,-1,-1,-1\n'

            if visualize:
                plot_info(img_v, f'{vid_name}: {i + 1} / {len(img_names)}', font_size=1.2, font_thickness=2)
                if view_size is not None:
                    img_v = letterbox(img_v, view_size)[0]

                cv2.imshow(vid_name, img_v)
                keyboard_input = cv2.waitKey(0) & 0xff
                if keyboard_input == ord('q'):
                    break
                elif keyboard_input == 27:  # 27: esc
                    sys.exit()

            if vid_writer is not None:
                vid_writer.write(img_v)

        if visualize:
            cv2.destroyWindow(vid_name)

        # save tracking prediction for TrackEval
        if save_pred:
            track_save_dir = os.path.join(save_dir, f'{target_select}-{target_split}',
                                          REFERENCE_TRACKERS[select_tracker], 'data')  # for using TrackEval coda
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

    end_time = time.time()
    print(f'\nElapsed time: {end_time - start_time:.2f}')


def get_args():
    parser = argparse.ArgumentParser()

    # Arguments for MOT17/MOT20 dataset
    mot_root = '/home/jhc/Desktop/dataset/open_dataset/MOT'  # path to MOT dataset
    parser.add_argument('--mot_root', type=str, default=mot_root)

    target_select = 'MOT17'  # select in ['MOT17', 'MOT20']
    parser.add_argument('--target_select', type=str, default=target_select)

    target_split = 'train'  # select in ['train', 'test']
    parser.add_argument('--target_split', type=str, default=target_split)

    target_vid = None  # None: all videos, other numbers: target videos
    # for MOT17 train, select in [2, 4, 5, 9, 10, 11, 13]
    # for MOT17 test, select in [1, 3, 6, 7, 8, 12, 14]
    # for MOT20 train, select in [1, 2, 3, 5]
    # for MOT20 test, select in [4, 6, 7, 8]
    parser.add_argument('--target_vid', type=int, default=target_vid, nargs='+')

    target_det = ['FRCNN']  # for MOT17, select in ['DPM', 'FRCNN', 'SDP']
    parser.add_argument('--target_det', type=str, default=target_det, nargs='+')

    # Arguments for detector
    det_cfg_file = 'test_det'  # file name of target config in tracker_cfgs directory
    parser.add_argument('--det_cfg_file', type=str, default=det_cfg_file)
    parser.add_argument('--use_detector', action='store_true', default=True)
    parser.add_argument('--use_saved_detector_result', action='store_true', default=True)
    parser.add_argument('--detector_result_dir', type=str, default='yolox_x_byte_mot17')

    # Arguments for reference tracker
    parser.add_argument('--select_tracker', type=int, default=5)  # {1: 'byte', 2: 'oc', 3: 'sort', 4: 'deepsort', ...}

    # General arguments for inference
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--vis_progress_bar', action='store_true', default=True)
    parser.add_argument('--out_dir', type=str, default=f'{FILE.parents[0]}/runs_ref/{target_select}_{target_split}')
    parser.add_argument('--run_name', type=str, default='BYTE_origin')
    parser.add_argument('--vis_det', action='store_true', default=False)
    parser.add_argument('--vis_trk', action='store_true', default=False)
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
