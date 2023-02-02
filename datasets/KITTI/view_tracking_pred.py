# Visualize tracking label of KITTI dataset
import argparse
import os
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
from screeninfo import get_monitors


CLASSES = {'Car': 1, 'Van': 2, 'Truck': 3,
           'Pedestrian': 4, 'Person_sitting': 5, 'Person': 5, 'Cyclist': 6,
           'Tram': 7, 'Misc': 8, 'DontCare': 9}


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def parsing_kitti_pred(pred_path):
    with open(pred_path) as f:
        rets = [x.strip('\n').split() for x in f.readlines()]
        pred = defaultdict(list)
        for ret in rets:
            pred[int(ret[0])].append(ret[1:])
    return pred


def main(args):
    kitti_root = args.kitti_root
    target_vid = args.target_vid
    target_split = args.target_split
    pred_dirs = args.pred_dirs
    view_size = args.view_size

    vid_root = os.path.join(kitti_root, 'data_tracking_image_2', target_split, 'image_02')
    if not os.path.isdir(vid_root):
        raise Exception(f'Given target split "{target_split}" is wrong!')

    if target_vid is not None:
        vid_list = [x for x in sorted(os.listdir(vid_root)) if int(x) in target_vid]
    else:
        vid_list = sorted(os.listdir(vid_root))

    pred_list = [{x.split('.')[0]: x for x in os.listdir(pred_dir)} for pred_dir in pred_dirs.values()]

    tmp_monitor = get_monitors()[1]
    ms = int(tmp_monitor.height * 0.8), int(tmp_monitor.width * 0.8)
    view_size = view_size if view_size is not None else ms

    for vid_idx, vid_name in enumerate(vid_list):
        print(f"\n--- Processing {vid_idx + 1} / {len(vid_list)}'s video: {vid_name}")
        vid_dir = os.path.join(vid_root, vid_name)

        pred_files = [x[vid_name] for x in pred_list]
        pred_paths = [os.path.join(pred_dir, pred_file) for pred_dir, pred_file in zip(pred_dirs.values(), pred_files)]
        preds = [parsing_kitti_pred(pred_path) for pred_path in pred_paths]

        imgs = sorted(os.listdir(vid_dir))

        for i, img_name in tqdm(enumerate(imgs), total=len(imgs)):
            img_path = os.path.join(vid_dir, img_name)
            img = cv2.imread(img_path)
            plot_info(img, f'{vid_name}: {i + 1} / {len(imgs)}')

            pred_imgs = []
            for j, (tracker_name, pred) in enumerate(zip(pred_dirs.keys(), preds)):
                tmp_img = img.copy()
                plot_info(tmp_img, f'{tracker_name}: {vid_name}: {i + 1} / {len(imgs)}', font_size=2, font_thickness=2)
                bboxes = pred[i]
                for bbox in bboxes:
                    track_id = int(float(bbox[0]))
                    trk_color = colors(track_id, True)
                    cls_type = bbox[1]
                    xyxy = np.array(list(map(float, bbox[5: 9])))
                    cv2.rectangle(tmp_img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), trk_color, 1)
                    plot_label(tmp_img, xyxy, f'{str(track_id)}: {cls_type}', trk_color, font_size=0.4)
                pred_imgs.append(tmp_img)

            pred_img = np.vstack(pred_imgs)
            pred_img = letterbox(pred_img, view_size, auto=False)[0]

            if view_size is not None:
                img = letterbox(pred_img, view_size, auto=False)[0]

            cv2.imshow(vid_name, img)
            keyboard_input = cv2.waitKey(0) & 0xff
            if keyboard_input == ord('q'):
                break
            elif keyboard_input == 27:  # 27: esc
                import sys
                sys.exit()
        cv2.destroyWindow(vid_name)


def xywh2xyxy(xywh):
    xyxy = xywh.copy()
    xyxy[2] = xyxy[0] + xyxy[2]
    xyxy[3] = xyxy[1] + xyxy[3]
    return xyxy


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def plot_info(img, info, font_size=1, font_thickness=1):
    label_size = cv2.getTextSize(info, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)[0]
    cv2.rectangle(img, (0, 0), (label_size[0] + 10, label_size[1] * 2), [0, 0, 0], -1)
    cv2.putText(img, info, (5, int(label_size[1] * 1.5))
                , cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)


def plot_label(img, xyxy, label, color, font_size=0.4, font_thickness=1):
    txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)[0]
    txt_bk_color = [int(c * 0.7) for c in color]
    cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[0]) + txt_size[0] + 1, int(xyxy[1]) - int(txt_size[1] * 1.5)),
                  txt_bk_color, -1)
    cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - int(txt_size[1] * 0.4)),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness)


def parse_args():
    parser = argparse.ArgumentParser()

    # Arguments for KITTI tracking dataset
    kitti_root = '/home/jhc/Desktop/dataset/open_dataset/KITTI'
    parser.add_argument('--kitti_root', type=str, default=kitti_root)

    target_split = 'testing'  # ['training', 'testing']
    parser.add_argument('--target_split', type=str, default=target_split)

    target_vid = None  # None: all videos, other numbers: target videos
    parser.add_argument('--target_vid', type=int, default=target_vid, nargs='+')

    target_cls = None  # None: all classes, other numbers: target classes
    parser.add_argument('--target_cls', type=int, default=target_cls, nargs='+')

    # Arguments for tracking predictions
    pred_dirs = {
        'ocsort': '/home/jhc/PycharmProjects/pythonProject/SORT_FAMILY/OC_SORT/kitti_test',
        'ocsort2': '/home/jhc/PycharmProjects/pythonProject/SORT_FAMILY/OC_SORT/exps/permatrack_kitti_test'
    }
    parser.add_argument('--pred_dirs', type=str, default=pred_dirs, nargs='+')

    # General Arguments
    parser.add_argument("--view-size", type=int, default=[720, 1280])

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_args()
    main(opt)
