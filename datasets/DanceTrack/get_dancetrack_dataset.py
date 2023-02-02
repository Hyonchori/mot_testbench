import os
from typing import List
from collections import defaultdict


DANCE_SPLIT = ['train', 'val', 'test']
DANCE_VID = {
    'train': [1, 2, 6, 8, 12, 15, 16, 20, 23, 24, 27, 29, 32, 33, 37, 39, 44, 45, 49,
              51, 52, 53, 55, 57, 61, 62, 66, 68, 69, 72, 74, 75, 80, 82, 83, 86, 87, 96, 98, 99],  # total 40
    'val': [4, 5, 7, 10, 14, 18, 19, 25, 26, 30, 34, 35, 41, 43, 47, 58, 63, 65, 73,
            77, 79, 81, 90, 94, 97],  # total 25,
    'test': [3, 9, 11, 13, 17, 21, 22, 28, 31, 36, 38, 40, 42, 46, 48, 50, 54, 56, 59,
             60, 64, 67, 70, 71, 76, 78, 84, 85, 88, 89, 91, 92, 93, 95, 100]  # total 35
}


def get_dance_videos(
        dance_root: str,  # path to DanceTrack dataset
        target_split: str,  # select in DANCE_SPLIT
        target_vid: List[int]  # select in DANCE_VID
):
    vid_root = os.path.join(dance_root, target_split)
    if not os.path.isdir(vid_root):
        raise ValueError(f'Given arguments are wrong!: {dance_root}, {target_split}')

    vid_list = sorted(os.listdir(vid_root))
    if target_vid is not None:
        vid_list = [x for x in vid_list if int(x[-4:]) in target_vid]

    print(f'\nLoading videos from DanceTrack-{target_split}... ')
    print(f'\ttotal {len(vid_list)} videos are ready!')
    return vid_root, vid_list


def parsing_dance_detection(det_path: str):
    with open(det_path) as f:
        rets = [list(map(float, x.strip('\n').split(','))) for x in f.readlines()]
        dets = defaultdict(list)
        for ret in rets:
            dets[int(ret[0]) - 1].append(ret[2:7] + [0])
    return dets
