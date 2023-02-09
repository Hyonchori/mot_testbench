from types import SimpleNamespace


def make_bot_args():
    args = SimpleNamespace()

    args.device = 'gpu'
    args.mot20 = False

    args.track_high_thresh = 0.6
    args.track_low_thresh = 0.1
    args.new_track_thresh = 0.7
    args.track_buffer = 30
    args.match_thresh = 0.8
    args.aspect_ratio_thresh = 1.6
    args.min_box_area = 10

    args.cmc_method = 'none'
    args.name = 2
    args.ablation = None

    args.with_reid = True
    args.fast_reid_config = 'ref_trackers/botsort_tracker/fast_reid/configs/MOT17/sbs_S50.yml'
    args.fast_reid_weights = 'ref_trackers/botsort_tracker/fast_reid/weights/mot17_sbs_S50.pth'
    args.proximity_thresh = 0.5
    args.appearance_thresh = 0.25

    return args
