import numpy as np


class Controller:

    def __init__(self, controllability, A, B, deadbeat_control, state_estimate, init_control_sequence):
        self.A = A
        self.B = B
        self.controllability = controllability
        self.deadbeat_control = deadbeat_control
        self.phi = A + B @ self.deadbeat_control

        self.state_estimate = state_estimate
        self.control_sequence = init_control_sequence

        self.actuator_control_sequence = init_control_sequence

    def update_state_estimate(self):
        u = self.actuator_control_sequence[0]
        self.actuator_control_sequence = np.roll(self.actuator_control_sequence, -1)
        self.actuator_control_sequence[-1] = 0
        self.state_estimate = self.A @ self.state_estimate + self.B * u

    def update_control_sequence(self):
        self.control_sequence = np.reshape([self.deadbeat_control
                                            @ np.linalg.matrix_power(self.phi, i)
                                            @ self.state_estimate
                                            for i in range(self.controllability)], (self.controllability,))
