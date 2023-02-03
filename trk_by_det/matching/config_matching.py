from .matching_strategies import associate


def get_matching_fn(cfg):
    matching_dict = {
        'basic': associate
    }
    if cfg.type_matching not in matching_dict:
        raise KeyError(f'Given type_matching "{cfg.type_matching}" is not in {matching_dict}')

    return matching_dict[cfg.type_matching]
