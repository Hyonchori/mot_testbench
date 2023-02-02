import numpy as np
from .base_kalman_filter import BaseKalmanFilter


kalman_filter_config = {}


def init_kalman_filter_config(cfg):
    if cfg.type_kalman_filter == 'deep_sort':
        A = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        H = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1]
        ], dtype=np.float32)
        Q = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        R = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    elif cfg.type_kalman_filter == 'sort':
        A = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        H = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        Q = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        R = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    elif cfg.type_kalman_filter == 'custom':
        A = cfg.system_matrix
        H = cfg.projection_matrix
        Q = cfg.system_noise
        R = cfg.measurement_noise

    else:
        A = cfg.system_matrix
        H = cfg.projection_matrix
        Q = cfg.system_noise
        R = cfg.measurement_noise

    '''configure covariance initialization function'''
    if cfg.type_state == 'cpah':
        def init_cov_func(pos_weight, vel_weight, height: float = None, state_dim: int = None):
            # state: [cx, cy, aspect_ratio, height, cx', cy', aspect_ratio', height']
            std = [
                2 * pos_weight * height,
                2 * pos_weight * height,
                1e-2,
                2 * pos_weight * height,
                10 * vel_weight * height,
                10 * vel_weight * height,
                1e-5,
                10 * vel_weight * height
            ]
            cov = np.diag(np.square(std), dtype=np.float32)
            return cov

    elif cfg.type_state == 'cpwh':
        def init_cov_func(pos_weight, vel_weight, height: float = None, state_dim: int = None):
            # state: [cx, cy, width, height, cx', cy', width', height']
            std = [
                2 * pos_weight * height,
                2 * pos_weight * height,
                2 * pos_weight * height,
                2 * pos_weight * height,
                10 * vel_weight * height,
                10 * vel_weight * height,
                10 * vel_weight * height,
                10 * vel_weight * height,
            ]
            cov = np.diag(np.square(std), dtype=np.float32)
            return cov

    elif cfg.type_state == 'cpsa':
        def init_cov_func(pos_weight, vel_weight, height: float = None, state_dim: int = None):
            # state: [cx, cy, space, aspect_ratio, cx', cy', width', height']
            cov = np.identity(state_dim, dtype=np.float32)
            return cov

    elif cfg.type_state == 'custom':
        def init_cov_func(pos_weight, vel_weight, height: float = None, state_dim: int = None):
            cov = np.identity(state_dim, dtype=np.float32)
            cov[4, 4] = 1000
            cov[5, 5] = 1000
            return cov

    else:
        def init_cov_func(pos_weight, vel_weight, height: float = None, state_dim: int = None):
            cov = np.identity(state_dim, dtype=np.float32)
            cov[4, 4] = 1000
            cov[5, 5] = 1000
            return cov

    '''configure prediction noise initialization function'''
    if cfg.type_state == 'cpah':
        def predict_noise_func(pos_weight, vel_weight, height: float = None, system_noise=None):
            # state: [cx, cy, aspect_ratio, height, cx', cy', aspect_ratio', height']
            std = [
                pos_weight * height,
                pos_weight * height,
                1e-2,
                pos_weight * height,
                vel_weight * height,
                vel_weight * height,
                1e-5,
                vel_weight * height
            ]
            predict_noise = np.diag(np.square(std), dtype=np.float32)
            return predict_noise

    elif cfg.type_state == 'cpwh':
        def predict_noise_func(pos_weight, vel_weight, height: float = None, system_noise=None):
            # state: [cx, cy, width, height, cx', cy', width', height']
            std = [
                pos_weight * height,
                pos_weight * height,
                pos_weight * height,
                pos_weight * height,
                vel_weight * height,
                vel_weight * height,
                vel_weight * height,
                vel_weight * height,
            ]
            predict_noise = np.diag(np.square(std), dtype=np.float32)
            return predict_noise

    elif cfg.type_state == 'custom':
        def predict_noise_func(pos_weight, vel_weight, height: float = None, system_noise=None):
            predict_noise = system_noise
            return predict_noise

    else:
        def predict_noise_func(pos_weight, vel_weight, height: float = None, system_noise=None):
            predict_noise = system_noise
            return predict_noise

    '''configure projection noise initialization function'''
    if cfg.type_state == 'cpah':
        def project_noise_func(pos_weight, height: float = None, measurement_noise=None):
            # state: [cx, cy, aspect_ratio, height, cx', cy', aspect_ratio', height']
            std = [
                pos_weight * height,
                pos_weight * height,
                1e-1,
                pos_weight * height,
            ]
            project_noise = np.diag(np.square(std), dtype=np.float32)
            return project_noise

    elif cfg.type_state == 'cpwh':
        def project_noise_func(pos_weight, height: float = None, measurement_noise=None):
            # state: [cx, cy, width, height, cx', cy', width', height']
            std = [
                pos_weight * height,
                pos_weight * height,
                pos_weight * height,
                pos_weight * height,
            ]
            project_noise = np.diag(np.square(std), dtype=np.float32)
            return project_noise

    elif cfg.type_state == 'custom':
        def project_noise_func(pos_weight, height: float = None, measurement_noise=None):
            project_noise = measurement_noise
            return project_noise

    else:
        def project_noise_func(pos_weight, height: float = None, measurement_noise=None):
            project_noise = measurement_noise
            return project_noise

    kalman_filter_config['system_matrix'] = A
    kalman_filter_config['projection_matrix'] = H
    kalman_filter_config['system_noise'] = Q
    kalman_filter_config['measurement_noise'] = R
    kalman_filter_config['init_cov_func'] = init_cov_func
    kalman_filter_config['predict_noise_func'] = predict_noise_func
    kalman_filter_config['project_noise_func'] = project_noise_func


def get_kalman_filter(cfg, init_measure: np.ndarray):
    return BaseKalmanFilter(
        system_matrix=kalman_filter_config['system_matrix'],
        projection_matrix=kalman_filter_config['projection_matrix'],
        system_noise=kalman_filter_config['system_noise'],
        measurement_noise=kalman_filter_config['measurement_noise'],
        init_measure=init_measure,
        init_cov_func=kalman_filter_config['init_cov_func'],
        predict_noise_func=kalman_filter_config['predict_noise_func'],
        project_noise_func=kalman_filter_config['project_noise_func'],
        std_weight_position=cfg.std_weight_position,
        std_weight_velocity=cfg.std_weight_velocity,
        is_nsa=cfg.is_nsa,
        use_oos=cfg.use_oos
    )




