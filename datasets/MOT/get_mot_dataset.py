import os
from typing import List
from collections import defaultdict


MOT_SELECTION = ['MOT17', 'MOT20']
MOT_SPLIT = ['train', 'test']
MOT_VID = {
    'MOT17':
        {'train': [2, 4, 5, 9, 10, 11, 13],  # total 7
         'test': [1, 3, 6, 7, 8, 12, 14]},  # total 7
    'MOT20':
        {'train': [1, 2, 3, 5],  # total 4
         'test': [4, 6, 7, 8]}  # total 4
}
MOT17_DET = ['DPM', 'FRCNN', 'SDP']
MOT_CLASSES = {0: 'pedestrian', 1: 'person_on_vehicle', 2: 'car', 3: 'bicycle', 4: 'motorbike',
               5: 'non_motorized_vehicle', 6: 'static_person', 7: 'distractor', 8: 'occluder',
               9: 'occluder_on_the_ground', 10: 'occluder_full', 11: 'reflection', 12: 'crowd'}


def get_mot_videos(
        mot_root: str,  # path to MOT dataset
        target_select: str,  # select in MOT_SELECTION
        target_split: str,  # select in MOT_SPLIT
        target_vid: List[int] = None,  # select in MOT_VID
        target_det: List[str] = None  # for MOT17, select in MOT17_DET
):
    vid_root = os.path.join(mot_root, target_select, target_split)
    if not os.path.isdir(vid_root):
        raise ValueError(f'Given arguments are wrong!: {mot_root}, {target_select}, {target_split}')

    vid_list = sorted(os.listdir(vid_root))
    if target_vid is not None:
        vid_list = [x for x in vid_list if int(x.split('-')[1]) in target_vid]

    if target_select == 'MOT17' and target_det is not None:
        vid_list = [x for x in vid_list if x.split('-')[-1] in target_det]
        remain_dets = [x for x in MOT17_DET if not x in target_det]
    elif target_select == 'MOT17' and target_det is None:
        vid_list = [x for x in vid_list if x.split('-')[-1] == 'FRCNN']
        remain_dets = ['DPM', 'SDP']
    else:
        remain_dets = []

    print(f'\nLoading videos from {target_select}-{target_split}... ')
    print(f'\ttotal {len(vid_list)} videos are ready!')
    return vid_root, vid_list, remain_dets


def parsing_mot_gt(gt_path: str):
    with open(gt_path) as f:
        rets = [x.strip('\n').split(',') for x in f.readlines()]
        gt = defaultdict(list)
        for ret in rets:
            gt[int(ret[0]) - 1].append(ret[1:])
    return gt


def parsing_mot_detection(det_path: str):
    with open(det_path) as f:
        rets = [list(map(float, x.strip('\n').split(','))) for x in f.readlines()]
        dets = defaultdict(list)
        for ret in rets:
            dets[int(ret[0]) - 1].append(ret[2:7] + [0])
    return dets
