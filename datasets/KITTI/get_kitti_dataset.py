import os
from typing import List
from collections import defaultdict


KITTI_SPLIT = ['training', 'testing']
KITTI_VID = {
    'training': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'testing': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28]
}
KITTI_CLASSES = {'Car': 1, 'Van': 2, 'Truck': 3,
                 'Pedestrian': 4, 'Person_sitting': 5, 'Person': 5, 'Cyclist': 6,
                 'Tram': 7, 'Misc': 8, 'DontCare': 9}


def get_kitti_videos(
        kitti_root: str,  # path to KITTI dataset
        target_split: str,  # select in KITTI_SPLIT
        target_vid: List[int]  # select in KITTI_VID
):
    vid_root = os.path.join(kitti_root, 'data_tracking_image_2', target_split, 'image_02')
    if not os.path.isdir(vid_root):
        raise Exception(f'Given target split "{target_split}" is wrong!')

    if target_vid is not None:
        vid_list = [x for x in sorted(os.listdir(vid_root)) if int(x) in target_vid]
    else:
        vid_list = sorted(os.listdir(vid_root))

    print(f'\nLoading videos from KITTI-{target_split}... ')
    print(f'\ttotal {len(vid_list)} videos are ready!')
    return vid_root, vid_list


def parsing_kitti_det(det_path: str):
    with open(det_path) as f:
        rets = [x.strip('\n').split() for x in f.readlines()]
        det = defaultdict(list)
        for ret in rets:
            det[int(ret[0])].append(ret[1:])
    return det
