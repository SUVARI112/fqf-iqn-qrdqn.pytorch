import numpy as np
from fqf_iqn_qrdqn.environment.kalman_filter import KalmanFilter
from fqf_iqn_qrdqn.environment.controller import Controller


class Plant:

    def __init__(self, dim, controllability, A, B, C, Qw, Qv, deadbeat_control):

        self.dim = dim
        self.controllability = controllability

        self.A = A
        self.B = B
        self.C = C
        self.Qw = Qw
        self.Qv = Qv
        self.deadbeat_control = deadbeat_control
        
        self.kalman_filter = KalmanFilter(dim, A, B, C, Qw, Qv)

        self.controller = Controller(controllability,
                                     A,
                                     B,
                                     deadbeat_control,
                                     [[0], [0]],
                                     np.zeros(controllability))

        self.x = [[0], [0]]
        self.y = [[0], [0]]

        self.control_sequence = np.zeros(controllability)

        self.current_command = self.control_sequence[0]

    def update(self):
        self.current_command = self.control_sequence[0]
        self.control_sequence = np.roll(self.control_sequence, -1)
        self.control_sequence[-1] = 0

        w = np.random.multivariate_normal(np.zeros(self.dim), self.Qw).reshape(2, 1)
        self.x = self.A @ self.x + self.B * self.current_command + w

        v = np.random.multivariate_normal(np.zeros(self.dim), self.Qw).reshape(2, 1)
        self.y = self.C @ self.x + v

        self.kalman_filter.update(self.current_command, self.y)

    def reset(self):
        self.x = [[0], [0]]
        self.y = [[0], [0]]
        self.control_sequence = np.zeros(self.controllability)
        self.current_command = self.control_sequence[0]
        self.kalman_filter.reset()

        self.controller = Controller(self.controllability,
                                     self.A,
                                     self.B,
                                     self.deadbeat_control,
                                     [[0], [0]],
                                     np.zeros(self.controllability))