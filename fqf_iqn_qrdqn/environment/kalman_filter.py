import numpy as np


class KalmanFilter:

    def __init__(self, dim, A, B, C, Qw, Qv):
        self.dim = dim

        self.A = A
        self.B = B
        self.C = C
        self.Qw = Qw
        self.Qv = Qv

        self.prior_state_estimation = [[0], [0]]
        self.posterior_state_estimation = [[0], [0]]

        self.P_prior = np.identity(dim)
        self.K = self.P_prior @ self.C.T @ np.linalg.inv(self.C @ self.P_prior @ self.C.T + self.Qv)
        self.P_posterior = (np.identity(self.dim) - self.K @ self.C) @ self.P_prior

    def update(self, u, y):
        self.prior_state_estimation = self.A @ self.posterior_state_estimation + self.B * u
        self.P_prior = self.A @ self.P_posterior @ self.A.T + self.Qw
        self.K = self.P_prior @ self.C.T @ np.linalg.inv(self.C @ self.P_prior @ self.C.T + self.Qv)
        self.posterior_state_estimation = (self.prior_state_estimation
                                           + self.K @ (y - self.C @ self.prior_state_estimation))
        self.P_posterior = (np.identity(self.dim) - self.K @ self.C) @ self.P_prior

    def reset(self):
        self.prior_state_estimation = [[0], [0]]
        self.posterior_state_estimation = [[0], [0]]

        self.P_prior = np.identity(self.dim)
        self.K = self.P_prior @ self.C.T @ np.linalg.inv(self.C @ self.P_prior @ self.C.T + self.Qv)
        self.P_posterior = (np.identity(self.dim) - self.K @ self.C) @ self.P_prior
