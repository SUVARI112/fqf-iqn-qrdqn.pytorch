import copy
import random
from math import factorial as fact
import numpy as np
from itertools import combinations, permutations
from gym import spaces

class Environment():

    def __init__(self, plants, no_of_channels, uplink_coefficients, downlink_coefficients, controllability):
        self.plants = plants
        self.controllability = controllability
        self.N = len(plants)
        self.M = no_of_channels
        self.uplink_coefficients = uplink_coefficients
        self.downlink_coefficients = downlink_coefficients
        self.state = np.ones((2 * self.N, controllability + 1), dtype=int)
        # self.action_list = self.generate_combinations_and_permutations(self.M, self.N)
        self.action_list = self.generate_permutations_no_repetition(self.M, self.N)
        self.downlink_uplink_indicator = np.ones(self.N, dtype=int)

        self.S_x = np.identity(plants[0].dim)
        self.S_u = 1
        self.old_betas = np.zeros(self.N)

        self.seed_value = 112

        # Define minimum observation (ones array)
        minimum_state = np.ones((2 * self.N, controllability + 1), dtype=np.int32)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=minimum_state,
            high=1e10 * np.ones_like(minimum_state),
            dtype=np.int64
        )
        
        # Define action space (discrete actions from 0 to 33 based on your TF implementation)
        self.action_space = spaces.Discrete(len(self.action_list))  # 0 to 33 inclusive
        print("Number of Actions: ", len(self.action_list))
        print("Actions: ", self.action_list)

        # logging info
        self.total_cost = 0
        self.step_counter = 1

    def close(self):
        self.reset()
        pass

    def seed(self, seed):
        self.seed_value = seed
        random.seed(self.seed_value)
        np.random.seed(self.seed_value) 

    def step(self, action):

        # update controllers state estimate
        for plant in self.plants:
            plant.controller.update_state_estimate()

        # choose links
        action = self.action_list[action]
        links = action * self.downlink_uplink_indicator[action-1]
        # print(f"choosen Links: {links}")
        # uplink
        betas = np.zeros(self.N)
        link_no = 0
        # print(links)
        for link in links:
            if link > 0:
                channel_coefficient = self.uplink_coefficients[link_no, link - 1]
                if random.random() < channel_coefficient:
                    plant = self.plants[link - 1]
                    plant.controller.state_estimate = plant.kalman_filter.posterior_state_estimation
                    betas[link - 1] = 1
            link_no += 1

        # calculate control commands
        for plant in self.plants:
            plant.controller.update_control_sequence()

        # downlink
        gammas = np.zeros(self.N)
        link_no = 0
        for link in links:
            if link < 0:
                channel_coefficient = self.downlink_coefficients[link_no, - link - 1]
                if random.random() < channel_coefficient:
                    plant = self.plants[- link - 1]
                    plant.controller.actuator_control_sequence = plant.controller.control_sequence
                    plant.control_sequence = plant.controller.control_sequence
                    gammas[(- link - 1)] = 1
            link_no += 1

        # updating uplink downlink indicators
        for b in range(self.N):
            s = self.downlink_uplink_indicator[b]
            beta = betas[b]
            gamma = gammas[b]
            if s == 1 and beta == 1:
                self.downlink_uplink_indicator[b] = -1
            elif s == -1 and gamma == 1:
                self.downlink_uplink_indicator[b] = 1

        # update age of information
        old_state = copy.deepcopy(self.state)
        for plant_no in range(self.N):
            for j in range(self.controllability + 1):
                # tau update
                if j == 0:
                    if self.old_betas[plant_no]:
                        self.state[plant_no][j] = 1
                    else:
                        self.state[plant_no][j] = old_state[plant_no][j] + 1
                else:
                    if gammas[plant_no]:
                        self.state[plant_no][j] = old_state[plant_no][j - 1]
                    else:
                        self.state[plant_no][j] = old_state[plant_no][j]
                # eta update
                if j == 0:
                    if gammas[plant_no]:
                        self.state[self.N + plant_no][j] = 1
                    else:
                        self.state[self.N + plant_no][j] = old_state[self.N + plant_no][j] + 1
                else:
                    if gammas[plant_no]:
                        self.state[self.N + plant_no][j] = old_state[self.N + plant_no][j - 1]
                    else:
                        self.state[self.N + plant_no][j] = old_state[self.N + plant_no][j]

        
        self.old_betas = betas

        # update plants
        for plant in self.plants:
            plant.update()

        # update costs
        empirical_cost = 0
        expected_cost = 0
        plant_no = 0
        for plant in self.plants:
            empirical_cost += self.empirical_cost_calculation(plant) / self.N
            expected_cost += self.expected_cost_calculation(plant_no) / self.N
            plant_no += 1
        # print("this is self._state")
        # print(self._state)
        scaling_factor = 1000
        if empirical_cost < 1:
            cost = 1
        else:
            cost = scaling_factor*np.log(empirical_cost)

        self.step_counter += 1
        self.total_cost += empirical_cost

        return self.state.copy(), -1*cost, False, {}

    def reset(self):
        self.state = np.ones((2 * self.N, self.controllability + 1), dtype=int)
        for plant in self.plants:
            plant.reset()
        
        # log the episode info
        self.save_evaluation_data(file=f"log_evaluation_data_{self.M}-{self.N}.npy")
        self.total_cost = 0
        self.step_counter = 0

        return self.state

    def random_reset(self):
        span = np.random.randint(21, 100)
        success = np.random.randint(20, span)
        self.state = self.initialize_random_state(self.controllability, span, success)
        for plant in self.plants:
            plant.reset()

    def initialize_random_state(self, controllability, span, success):
        init_state = np.zeros((2 * 3, controllability + 1))
        for i in range(3):
            a = np.append(self.create_array_with_ones(span, success), 1)
            b = np.append(self.create_array_with_ones(span, success), 1)
            random_tau = self.calculate_correct_differences(a, b)[-3:]
            random_eta = self.differences_between_ones(b)[-3:]
            init_state[i, :] = random_tau
            init_state[i + 3, :] = random_eta
        
        return init_state.astype(int)

    @staticmethod
    def calculate_correct_differences(arr1, arr2):
        indices1 = np.where(arr1 == 1)[0]
        indices2 = np.where(arr2 == 1)[0]

        differences = []
        for index2 in indices2[::-1]:  # Start from the rightmost '1' in arr2
            left_indices1 = indices1[indices1 < index2]
            if left_indices1.size > 0:
                closest_left_index = left_indices1[-1]
                differences.append(index2 - closest_left_index)
            else:
                # If no '1' to the left in arr1, ignore this one in arr2
                continue

        return differences[::-1]  # Reverse to match the order of ones in arr2

    @staticmethod
    def differences_between_ones(arr):
        # Find the indices of ones
        indices = np.where(arr == 1)[0]

        # Calculate differences between consecutive indices
        differences = np.diff(indices)

        return differences.tolist()

    @staticmethod
    def create_array_with_ones(length, ones_count):
        # Ensure the number of ones does not exceed the array length
        if ones_count > length:
            raise ValueError("Number of ones cannot be greater than the array length.")

        # Create an array of zeros
        arr = np.zeros(length, dtype=int)

        # Randomly assign ones
        ones_indices = random.sample(range(length), ones_count)
        arr[ones_indices] = 1

        return arr

    def empirical_cost_calculation(self, plant):
        return plant.x.T @ self.S_x @ plant.x + plant.current_command * self.S_u * plant.current_command

    def expected_cost_calculation(self, plant_no):
        plant = self.plants[plant_no]
        A = plant.A
        B = plant.B
        C = plant.C
        Qv = plant.Qv
        Qw = plant.Qw
        K = plant.controller.deadbeat_control
        K_hat = plant.kalman_filter.K
        P = plant.kalman_filter.P_posterior
        taus = self.state[plant_no]
        etas = self.state[self.N + plant_no]
        tau = self.state[plant_no][0]
        eta = self.state[self.N + plant_no][0]
        v = self.controllability

        Z = (np.identity(plant.dim) - K_hat @ C) @ A

        D = np.add.reduce([
            np.linalg.matrix_power(A + B @ K, sum([etas[m] for m in range(j - 1)]))
            @ (np.linalg.matrix_power(A, etas[j - 1]) - np.linalg.matrix_power(A + B @ K, etas[j - 1]))
            @ np.linalg.matrix_power(A, taus[j])
            @ np.linalg.matrix_power(Z, self.delta(plant_no, j, v))
            for j in range(1, v + 1)
        ])

        V = (D @ P @ D.T
             + np.add.reduce([np.add.reduce([self.E_hatschek(plant_no, n, i)
                                             @ self.E_hatschek(plant_no, n, i).T
                                             for i in range(self.delta(plant_no, n, n + 1))])
                              for n in range(1, v)])
             + np.add.reduce([np.linalg.matrix_power(A, i)
                              @ Qw
                              @ np.linalg.matrix_power(A, i).T
                              for i in range(eta + taus[1])])
             + np.add.reduce([np.add.reduce([self.F_hatschek(plant_no, n, i)
                                             @ Qv
                                             @ self.F_hatschek(plant_no, n, i).T
                                             for i in range(self.delta(plant_no, n, n + 1))])
                              for n in range(1, v)])
             )

        D_wave = D - np.linalg.matrix_power(A, tau) @ np.linalg.matrix_power(Z, self.delta(plant_no, 0, v))

        V_hat = (D_wave @ P @ D_wave.T
                 + np.add.reduce([np.add.reduce([self.E_dot(plant_no, n, i)
                                                 @ self.E_dot(plant_no, n, i).T
                                                 for i in range(self.delta(plant_no, n, n + 1))])
                                  for n in range(1, v)])
                 + np.add.reduce([(np.linalg.matrix_power(A, tau + i) - np.linalg.matrix_power(A, tau)
                                   @ np.linalg.matrix_power(Z, i)
                                   @ (np.identity(plant.dim) - K_hat @ C))
                                  @ Qw
                                  @ (np.linalg.matrix_power(A, tau + i) - np.linalg.matrix_power(A, tau)
                                     @ np.linalg.matrix_power(Z, i)
                                     @ (np.identity(plant.dim) - K_hat @ C)).T
                                  for i in range(eta + taus[1])])
                 + np.add.reduce([np.add.reduce([self.F_dot(plant_no, n, i)
                                                 @ Qv
                                                 @ self.F_dot(plant_no, n, i).T
                                                 for i in range(self.delta(plant_no, n, n + 1))])
                                  for n in range(1, v)])
                 + np.add.reduce([(np.linalg.matrix_power(A, tau) @ np.linalg.matrix_power(Z, i) @ K_hat)
                                  @ Qw
                                  @ (np.linalg.matrix_power(A, tau) @ np.linalg.matrix_power(Z, i) @ K_hat).T
                                  for i in range(1, self.delta(plant_no, 0, 1))])
                 )

        J_x = np.trace(self.S_x @ V)

        J_u = np.trace((K @ np.linalg.matrix_power(A + B @ K, eta - 1)).T * self.S_u
                       @ (K @ np.linalg.matrix_power(A + B @ K, eta - 1))
                       @ V_hat)

        return J_x + J_u

    def delta(self, plant_no, i, j):
        taus = self.state[plant_no]
        etas = self.state[self.N + plant_no]
        # if taus[j] > 30:
        #     print(self.state)
        #     print("j", j)
        #     print(taus[j])
        return sum(etas[i:j]) + taus[j] - taus[i]

    def E_hatschek(self, plant_no, n, i):
        plant = self.plants[plant_no]
        A = plant.A
        B = plant.B
        C = plant.C
        K = plant.controller.deadbeat_control
        K_hat = plant.kalman_filter.K
        taus = self.state[plant_no]
        etas = self.state[self.N + plant_no]

        Z = (np.identity(plant.dim) - K_hat @ C) @ A

        E_hatschek = ((np.linalg.matrix_power(A + B @ K, sum([etas[m] for m in range(n)]))
                       @ np.linalg.matrix_power(A, taus[n] + i))
                      + np.add.reduce([np.linalg.matrix_power(A + B @ K, sum([etas[m] for m in range(j - 1)]))
                                       @ (np.linalg.matrix_power(A, etas[j - 1])
                                          - np.linalg.matrix_power(A + B @ K, etas[j - 1]))
                                       @ np.linalg.matrix_power(A, taus[j])
                                       @ np.linalg.matrix_power(Z, i + self.delta(plant_no, j, n))
                                       @ (np.identity(plant.dim) - K_hat @ C)
                                       for j in range(1, n + 1)]))

        return E_hatschek

    def F_hatschek(self, plant_no, n, i):
        plant = self.plants[plant_no]
        A = plant.A
        B = plant.B
        C = plant.C
        K = plant.controller.deadbeat_control
        K_hat = plant.kalman_filter.K
        taus = self.state[plant_no]
        etas = self.state[self.N + plant_no]

        Z = (np.identity(plant.dim) - K_hat @ C) @ A

        F_hatschek = np.add.reduce([np.linalg.matrix_power(A + B @ K, sum([etas[m] for m in range(j - 1)]))
                                    @ (np.linalg.matrix_power(A, etas[j - 1])
                                       - np.linalg.matrix_power(A + B @ K, etas[j - 1]))
                                    @ np.linalg.matrix_power(A, taus[j])
                                    @ np.linalg.matrix_power(Z, i + self.delta(plant_no, j, n))
                                    @ K_hat
                                    for j in range(1, n + 1)])

        return F_hatschek

    def E_dot(self, plant_no, n, i):
        plant = self.plants[plant_no]
        A = plant.A
        C = plant.C
        K_hat = plant.kalman_filter.K
        taus = self.state[plant_no]
        tau = self.state[plant_no][0]
        v = self.controllability

        Z = (np.identity(plant.dim) - K_hat @ C) @ A

        E_dot = (self.E_hatschek(plant_no, n, i)
                 - (np.linalg.matrix_power(A, tau)
                    @ np.linalg.matrix_power(Z, self.delta(plant_no, 0, v) - taus[n] + i)
                    @ (np.identity(plant.dim) - K_hat @ C)))

        return E_dot

    def F_dot(self, plant_no, n, i):
        plant = self.plants[plant_no]
        A = plant.A
        C = plant.C
        K_hat = plant.kalman_filter.K
        tau = self.state[plant_no][0]
        v = self.controllability

        Z = (np.identity(plant.dim) - K_hat @ C) @ A

        F_dot = (self.F_hatschek(plant_no, n, i)
                 + (np.linalg.matrix_power(A, tau)
                    @ np.linalg.matrix_power(Z, self.delta(plant_no, 0, v) + i)
                    @ K_hat))

        return F_dot
    
    @staticmethod
    def combinatorial_sum(M, N):
        # Function to calculate combinations
        def combinations(n, r):
            return fact(n) // (fact(r) * fact(n - r))

        # Function to calculate permutations, returns 0 if m > N
        def permutations(n, r):
            return fact(n) // fact(n - r) if r <= n else 0

        # Summing up the products of combinations and permutations for each m
        total_sum = sum(combinations(M, m) * permutations(N, m) for m in range(M + 1))

        return total_sum
    
    def save_evaluation_data(self, file="log_evaluation_data.npy"):
        
        try:
            # Laden vorhandener Daten, falls vorhanden
            existing_data = np.load(file, allow_pickle=True).tolist()
        except FileNotFoundError:
            existing_data = []
        
        data = float(self.total_cost/self.step_counter)
        print(f"Average Empirical Cost: {data}")
        print(f"---------------------------------------------")
        
        existing_data.append(data)
        np.save(file, np.array(existing_data))

    @staticmethod
    def generate_combinations_and_permutations(M, N):
        all_permutations = []

        # Generating combinations from the set [1, 2, 3, ..., M]
        for r in range(M + 1):
            for combination in combinations(range(1, M + 1), r):
                # Appending zeros to make it N elements long
                combination_with_zeros = list(combination) + [0] * (N - r)

                # Generating all permutations of this combination
                for perm in set(permutations(combination_with_zeros)):
                    all_permutations.append(list(perm))

        return np.asarray(all_permutations)

    @staticmethod
    def generate_permutations_no_repetition(M, N):
        all_permutations = []
        
        # Taking M numbers from range 1 to N
        for combination in combinations(range(1, N + 1), M):
            # Generate all permutations of each combination
            for perm in permutations(combination):
                all_permutations.append(list(perm))
                
        return np.asarray(all_permutations)


class Environment_eval():

    def __init__(self, plants, no_of_channels, uplink_coefficients, downlink_coefficients, controllability):
        self.plants = plants
        self.controllability = controllability
        self.N = len(plants)
        self.M = no_of_channels
        self.uplink_coefficients = uplink_coefficients
        self.downlink_coefficients = downlink_coefficients
        self.state = np.ones((2 * self.N, controllability + 1), dtype=int)
        self.action_list = self.generate_combinations_and_permutations(self.M, self.N)
        self.downlink_uplink_indicator = np.ones(self.N, dtype=int)

        self.S_x = np.identity(plants[0].dim)
        self.S_u = 1
        self.old_betas = np.zeros(self.N)

        self.seed_value = 112

        # Define minimum observation (ones array)
        minimum_state = np.ones((2 * self.N, controllability + 1), dtype=np.int32)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=minimum_state,
            high=1e10 * np.ones_like(minimum_state),
            dtype=np.int64
        )
        
        # Define action space (discrete actions from 0 to 33 based on your TF implementation)
        self.action_space = spaces.Discrete(34)  # 0 to 33 inclusive

    @staticmethod
    def generate_combinations_and_permutations(M, N):
        all_permutations = []

        # Generating combinations from the set [1, 2, 3, ..., M]
        for r in range(M + 1):
            for combination in combinations(range(1, M + 1), r):
                # Appending zeros to make it N elements long
                combination_with_zeros = list(combination) + [0] * (N - r)

                # Generating all permutations of this combination
                for perm in set(permutations(combination_with_zeros)):
                    all_permutations.append(list(perm))

        return np.asarray(all_permutations)


    def close(self):
        self.reset()
        pass

    def seed(self, seed):
        self.seed_value = seed
        random.seed(self.seed_value)
        np.random.seed(self.seed_value) 

    def step(self, action):

        # update controllers state estimate
        for plant in self.plants:
            plant.controller.update_state_estimate()

        # choose links
        action = self.action_list[action]
        links = action * self.downlink_uplink_indicator
        # print(f"choosen Links: {links}")
        # uplink
        betas = np.zeros(self.N)
        link_no = 0
        # print(links)
        for link in links:
            if link > 0:
                channel_coefficient = self.uplink_coefficients[link_no, link - 1]
                if random.random() < channel_coefficient:
                    plant = self.plants[link - 1]
                    plant.controller.state_estimate = plant.kalman_filter.posterior_state_estimation
                    betas[link - 1] = 1
            link_no += 1

        # calculate control commands
        for plant in self.plants:
            plant.controller.update_control_sequence()

        # downlink
        gammas = np.zeros(self.N)
        link_no = 0
        for link in links:
            if link < 0:
                channel_coefficient = self.downlink_coefficients[link_no, - link - 1]
                if random.random() < channel_coefficient:
                    plant = self.plants[- link - 1]
                    plant.controller.actuator_control_sequence = plant.controller.control_sequence
                    plant.control_sequence = plant.controller.control_sequence
                    gammas[(- link - 1)] = 1
            link_no += 1

        # updating uplink downlink indicators
        for b in range(self.N):
            s = self.downlink_uplink_indicator[b]
            beta = betas[b]
            gamma = gammas[b]
            if s == 1 and beta == 1:
                self.downlink_uplink_indicator[b] = -1
            elif s == -1 and gamma == 1:
                self.downlink_uplink_indicator[b] = 1

        # update age of information
        old_state = copy.deepcopy(self.state)
        for plant_no in range(self.N):
            for j in range(self.controllability + 1):
                # tau update
                if j == 0:
                    if self.old_betas[plant_no]:
                        self.state[plant_no][j] = 1
                    else:
                        self.state[plant_no][j] = old_state[plant_no][j] + 1
                else:
                    if gammas[plant_no]:
                        self.state[plant_no][j] = old_state[plant_no][j - 1]
                    else:
                        self.state[plant_no][j] = old_state[plant_no][j]
                # eta update
                if j == 0:
                    if gammas[plant_no]:
                        self.state[self.N + plant_no][j] = 1
                    else:
                        self.state[self.N + plant_no][j] = old_state[self.N + plant_no][j] + 1
                else:
                    if gammas[plant_no]:
                        self.state[self.N + plant_no][j] = old_state[self.N + plant_no][j - 1]
                    else:
                        self.state[self.N + plant_no][j] = old_state[self.N + plant_no][j]

        
        self.old_betas = betas

        # update plants
        for plant in self.plants:
            plant.update()

        # update costs
        empirical_cost = 0
        expected_cost = 0
        plant_no = 0
        for plant in self.plants:
            empirical_cost += self.empirical_cost_calculation(plant) / self.N
            expected_cost += self.expected_cost_calculation(plant_no) / self.N
            plant_no += 1
        # print("this is self._state")
        # print(self._state)

        return self.state.copy(), empirical_cost, False, {}

    def reset(self):
        self.state = np.ones((2 * self.N, self.controllability + 1), dtype=int)
        for plant in self.plants:
            plant.reset()
        return self.state

    def random_reset(self):
        span = np.random.randint(21, 100)
        success = np.random.randint(20, span)
        self.state = self.initialize_random_state(self.controllability, span, success)
        for plant in self.plants:
            plant.reset()

    def initialize_random_state(self, controllability, span, success):
        init_state = np.zeros((2 * 3, controllability + 1))
        for i in range(3):
            a = np.append(self.create_array_with_ones(span, success), 1)
            b = np.append(self.create_array_with_ones(span, success), 1)
            random_tau = self.calculate_correct_differences(a, b)[-3:]
            random_eta = self.differences_between_ones(b)[-3:]
            init_state[i, :] = random_tau
            init_state[i + 3, :] = random_eta
        
        return init_state.astype(int)

    @staticmethod
    def calculate_correct_differences(arr1, arr2):
        indices1 = np.where(arr1 == 1)[0]
        indices2 = np.where(arr2 == 1)[0]

        differences = []
        for index2 in indices2[::-1]:  # Start from the rightmost '1' in arr2
            left_indices1 = indices1[indices1 < index2]
            if left_indices1.size > 0:
                closest_left_index = left_indices1[-1]
                differences.append(index2 - closest_left_index)
            else:
                # If no '1' to the left in arr1, ignore this one in arr2
                continue

        return differences[::-1]  # Reverse to match the order of ones in arr2

    @staticmethod
    def differences_between_ones(arr):
        # Find the indices of ones
        indices = np.where(arr == 1)[0]

        # Calculate differences between consecutive indices
        differences = np.diff(indices)

        return differences.tolist()

    @staticmethod
    def create_array_with_ones(length, ones_count):
        # Ensure the number of ones does not exceed the array length
        if ones_count > length:
            raise ValueError("Number of ones cannot be greater than the array length.")

        # Create an array of zeros
        arr = np.zeros(length, dtype=int)

        # Randomly assign ones
        ones_indices = random.sample(range(length), ones_count)
        arr[ones_indices] = 1

        return arr

    def empirical_cost_calculation(self, plant):
        return plant.x.T @ self.S_x @ plant.x + plant.current_command * self.S_u * plant.current_command

    def expected_cost_calculation(self, plant_no):
        plant = self.plants[plant_no]
        A = plant.A
        B = plant.B
        C = plant.C
        Qv = plant.Qv
        Qw = plant.Qw
        K = plant.controller.deadbeat_control
        K_hat = plant.kalman_filter.K
        P = plant.kalman_filter.P_posterior
        taus = self.state[plant_no]
        etas = self.state[self.N + plant_no]
        tau = self.state[plant_no][0]
        eta = self.state[self.N + plant_no][0]
        v = self.controllability

        Z = (np.identity(plant.dim) - K_hat @ C) @ A

        D = np.add.reduce([
            np.linalg.matrix_power(A + B @ K, sum([etas[m] for m in range(j - 1)]))
            @ (np.linalg.matrix_power(A, etas[j - 1]) - np.linalg.matrix_power(A + B @ K, etas[j - 1]))
            @ np.linalg.matrix_power(A, taus[j])
            @ np.linalg.matrix_power(Z, self.delta(plant_no, j, v))
            for j in range(1, v + 1)
        ])

        V = (D @ P @ D.T
             + np.add.reduce([np.add.reduce([self.E_hatschek(plant_no, n, i)
                                             @ self.E_hatschek(plant_no, n, i).T
                                             for i in range(self.delta(plant_no, n, n + 1))])
                              for n in range(1, v)])
             + np.add.reduce([np.linalg.matrix_power(A, i)
                              @ Qw
                              @ np.linalg.matrix_power(A, i).T
                              for i in range(eta + taus[1])])
             + np.add.reduce([np.add.reduce([self.F_hatschek(plant_no, n, i)
                                             @ Qv
                                             @ self.F_hatschek(plant_no, n, i).T
                                             for i in range(self.delta(plant_no, n, n + 1))])
                              for n in range(1, v)])
             )

        D_wave = D - np.linalg.matrix_power(A, tau) @ np.linalg.matrix_power(Z, self.delta(plant_no, 0, v))

        V_hat = (D_wave @ P @ D_wave.T
                 + np.add.reduce([np.add.reduce([self.E_dot(plant_no, n, i)
                                                 @ self.E_dot(plant_no, n, i).T
                                                 for i in range(self.delta(plant_no, n, n + 1))])
                                  for n in range(1, v)])
                 + np.add.reduce([(np.linalg.matrix_power(A, tau + i) - np.linalg.matrix_power(A, tau)
                                   @ np.linalg.matrix_power(Z, i)
                                   @ (np.identity(plant.dim) - K_hat @ C))
                                  @ Qw
                                  @ (np.linalg.matrix_power(A, tau + i) - np.linalg.matrix_power(A, tau)
                                     @ np.linalg.matrix_power(Z, i)
                                     @ (np.identity(plant.dim) - K_hat @ C)).T
                                  for i in range(eta + taus[1])])
                 + np.add.reduce([np.add.reduce([self.F_dot(plant_no, n, i)
                                                 @ Qv
                                                 @ self.F_dot(plant_no, n, i).T
                                                 for i in range(self.delta(plant_no, n, n + 1))])
                                  for n in range(1, v)])
                 + np.add.reduce([(np.linalg.matrix_power(A, tau) @ np.linalg.matrix_power(Z, i) @ K_hat)
                                  @ Qw
                                  @ (np.linalg.matrix_power(A, tau) @ np.linalg.matrix_power(Z, i) @ K_hat).T
                                  for i in range(1, self.delta(plant_no, 0, 1))])
                 )

        J_x = np.trace(self.S_x @ V)

        J_u = np.trace((K @ np.linalg.matrix_power(A + B @ K, eta - 1)).T * self.S_u
                       @ (K @ np.linalg.matrix_power(A + B @ K, eta - 1))
                       @ V_hat)

        return J_x + J_u

    def delta(self, plant_no, i, j):
        taus = self.state[plant_no]
        etas = self.state[self.N + plant_no]
        # if taus[j] > 30:
        #     print(self.state)
        #     print("j", j)
        #     print(taus[j])
        return sum(etas[i:j]) + taus[j] - taus[i]

    def E_hatschek(self, plant_no, n, i):
        plant = self.plants[plant_no]
        A = plant.A
        B = plant.B
        C = plant.C
        K = plant.controller.deadbeat_control
        K_hat = plant.kalman_filter.K
        taus = self.state[plant_no]
        etas = self.state[self.N + plant_no]

        Z = (np.identity(plant.dim) - K_hat @ C) @ A

        E_hatschek = ((np.linalg.matrix_power(A + B @ K, sum([etas[m] for m in range(n)]))
                       @ np.linalg.matrix_power(A, taus[n] + i))
                      + np.add.reduce([np.linalg.matrix_power(A + B @ K, sum([etas[m] for m in range(j - 1)]))
                                       @ (np.linalg.matrix_power(A, etas[j - 1])
                                          - np.linalg.matrix_power(A + B @ K, etas[j - 1]))
                                       @ np.linalg.matrix_power(A, taus[j])
                                       @ np.linalg.matrix_power(Z, i + self.delta(plant_no, j, n))
                                       @ (np.identity(plant.dim) - K_hat @ C)
                                       for j in range(1, n + 1)]))

        return E_hatschek

    def F_hatschek(self, plant_no, n, i):
        plant = self.plants[plant_no]
        A = plant.A
        B = plant.B
        C = plant.C
        K = plant.controller.deadbeat_control
        K_hat = plant.kalman_filter.K
        taus = self.state[plant_no]
        etas = self.state[self.N + plant_no]

        Z = (np.identity(plant.dim) - K_hat @ C) @ A

        F_hatschek = np.add.reduce([np.linalg.matrix_power(A + B @ K, sum([etas[m] for m in range(j - 1)]))
                                    @ (np.linalg.matrix_power(A, etas[j - 1])
                                       - np.linalg.matrix_power(A + B @ K, etas[j - 1]))
                                    @ np.linalg.matrix_power(A, taus[j])
                                    @ np.linalg.matrix_power(Z, i + self.delta(plant_no, j, n))
                                    @ K_hat
                                    for j in range(1, n + 1)])

        return F_hatschek

    def E_dot(self, plant_no, n, i):
        plant = self.plants[plant_no]
        A = plant.A
        C = plant.C
        K_hat = plant.kalman_filter.K
        taus = self.state[plant_no]
        tau = self.state[plant_no][0]
        v = self.controllability

        Z = (np.identity(plant.dim) - K_hat @ C) @ A

        E_dot = (self.E_hatschek(plant_no, n, i)
                 - (np.linalg.matrix_power(A, tau)
                    @ np.linalg.matrix_power(Z, self.delta(plant_no, 0, v) - taus[n] + i)
                    @ (np.identity(plant.dim) - K_hat @ C)))

        return E_dot

    def F_dot(self, plant_no, n, i):
        plant = self.plants[plant_no]
        A = plant.A
        C = plant.C
        K_hat = plant.kalman_filter.K
        tau = self.state[plant_no][0]
        v = self.controllability

        Z = (np.identity(plant.dim) - K_hat @ C) @ A

        F_dot = (self.F_hatschek(plant_no, n, i)
                 + (np.linalg.matrix_power(A, tau)
                    @ np.linalg.matrix_power(Z, self.delta(plant_no, 0, v) + i)
                    @ K_hat))

        return F_dot
    
    @staticmethod
    def combinatorial_sum(M, N):
        # Function to calculate combinations
        def combinations(n, r):
            return fact(n) // (fact(r) * fact(n - r))

        # Function to calculate permutations, returns 0 if m > N
        def permutations(n, r):
            return fact(n) // fact(n - r) if r <= n else 0

        # Summing up the products of combinations and permutations for each m
        total_sum = sum(combinations(M, m) * permutations(N, m) for m in range(M + 1))

        return total_sum