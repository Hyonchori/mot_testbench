from types import SimpleNamespace


def make_deepsort_args():
    args = SimpleNamespace()

    args.model_path = '/home/jhc/PycharmProjects/pythonProject/SORT_FAMILY/mot_testbench/trk_by_det/feature_extractor/simple_cnn/weights/ckpt.t7'
    args.track_thresh = 0.6
    args.vertical_thresh = 1.6
    args.min_box_area = 100

    return args
