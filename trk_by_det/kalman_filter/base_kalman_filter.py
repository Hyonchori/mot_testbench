import numpy as np


class BaseKalmanFilter:
    def __init__(
            self,  # m: state_dim, n: measurement_dim
            system_matrix: np.ndarray,  # (m, m)
            projection_matrix: np.ndarray,  # (n, m)
            system_noise: np.ndarray,  # (m, m)
            measurement_noise: np.ndarray,  # (n, n)
            init_measure: np.ndarray,  # (n, 1)
            init_cov_func,
            predict_noise_func,
            project_noise_func,
            std_weight_position: float = 1. / 20,
            std_weight_velocity: float = 1. / 160,
            is_nsa: bool = False,
            use_oos: bool = False
    ):
        # matrices used in kalman prediction and update
        self.A = system_matrix
        self.H = projection_matrix
        self.Q = system_noise
        self.R = measurement_noise
        self.actual_R = measurement_noise.copy()

        # init measurement and options
        self.z = init_measure
        self.init_cov_func = init_cov_func
        self.predict_noise_func = predict_noise_func
        self.project_noise_func = project_noise_func
        self.is_nsa = is_nsa
        self.use_oos = use_oos

        # initial state and covariance
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity
        self.x, self.x_cov = self.initialize_state()

    def initialize_state(self):
        x = np.zeros([self.A.shape[0], 1], dtype=np.float32)  # kalman state: (m, 1)
        x[:self.H.shape[0]] = self.z
        x_cov = self.init_cov_func(
            pos_weight=self._std_weight_position,
            vel_weight=self._std_weight_velocity,
            height=x[3][0],
            state_dim=x.shape[0]
        )  # kalman state covariance: (m, m)
        return x, x_cov

    def predict(self):
        # kalman predict: predict state(t) using state(t-1)
        self.x = np.matmul(self.A, self.x)
        Q = self.predict_noise_func(
            pos_weight=self._std_weight_position,
            vel_weight=self._std_weight_velocity,
            height=self.x[3][0],
            system_noise=self.Q
        )
        self.x_cov = np.linalg.multi_dot([self.A, self.x_cov, self.A.T]) + Q

    def measure(self, new_z, conf, apply_oos=False, time_since_update=None):
        # update measurement by matching
        if self.is_nsa:
            self.actual_R = self.R * (1. - conf)

        # apply Observation Online Smoothing for 'rematched tracks'
        if self.use_oos and apply_oos:
            dz = (new_z - self.z) / time_since_update
            frozen_z = self.z.copy()
            for i in range(1, time_since_update):
                self.z = frozen_z + dz * i
                self.update()

        self.z = new_z

    def project(self):
        # kalman projection: project state-space to measurement-space
        projected_x = np.matmul(self.H, self.x)
        R = self.project_noise_func(
            pos_weight=self._std_weight_position,
            vel_weight=self._std_weight_velocity,
            height=self.x[3][0],
            measurement_noise=self.actual_R
        )
        projected_x_cov = np.linalg.multi_dot([self.H, self.x_cov, self.H.T]) + R
        return projected_x, projected_x_cov

    def update(self):
        # kalman update: calculate present state(t) using prediction(t) and prior state(t-1)
        projected_x, projected_x_cov = self.project()
        K = np.linalg.multi_dot([self.x_cov, self.H.T, np.linalg.inv(projected_x_cov)])
        y = self.z - projected_x
        self.x = self.x + np.matmul(K, y)
        self.x_cov = np.matmul(np.identity(self.A.shape[0], dtype=np.float32) - np.matmul(K, self.H), self.x_cov)
