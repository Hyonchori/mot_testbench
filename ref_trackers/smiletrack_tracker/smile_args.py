from types import SimpleNamespace


def make_smile_args():
    args = SimpleNamespace()

    args.device = '0'
    args.mot20 = False

    args.cmc_method = 'file'
    args.use_cmc_file = True
    args.cmc_result_dir = '/home/jhc/PycharmProjects/pythonProject/SORT_FAMILY/BoT-SORT/tracker/GMC_files/MOTChallenge'

    args.track_high_thresh = 0.5
    args.track_low_thresh = 0.1
    args.new_track_thresh = 0.6
    args.track_buffer = 30
    args.match_thresh = 0.8
    args.aspect_ratio_thresh = 1.6
    args.min_box_area = 10
    args.fuse_score = False

    args.with_reid = True
    args.reid_weight_path = 'ref_trackers/smiletrack_tracker/pretrained/ver12.pt'
    args.proximity_thresh = 0.5
    args.appearance_thresh = 0.25

    return args
