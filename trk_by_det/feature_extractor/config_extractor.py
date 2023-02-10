from .simple_cnn.simple_cnn_wrapper import get_simple_cnn_extractor
from .fast_reid.fast_reid_wrapper import get_fast_reid_extractor
from .slm.slm_wrapper import get_slm_extractor


def get_extractor(cfg, device=None):
    extractor_dict = {
        'simple_cnn': get_simple_cnn_extractor,
        'fast_reid': get_fast_reid_extractor,
        'slm': get_slm_extractor
    }
    if cfg.type_extractor not in extractor_dict:
        raise ValueError(f'type_extractor should be one of {extractor_dict.keys()}, but given {cfg.type_extractor}')

    return extractor_dict[cfg.type_extractor](cfg, device)
