import numpy as np
import random
from cogn_bucb import BUCB


class CognitiveUser:
    def __init__(self, C, U, gamma, epsilon):
        np.random.seed(30)
        self.X = [0] * C
        self.T = [0] * C
        self.C = C
        self.U = U
        self.gamma = gamma
        self.epsilon = epsilon
        self.total_reward = 0
        self.r1 = np.random.randint(1, self.U)
        self.r2 = np.random.randint(self.U + 1, min(2 * self.U, self.C))
        self.random_array = [x for x in range(C)]
        random.shuffle(self.random_array)

    def first_time_choice(self, time):
        N1 = BUCB(time, self.r1, self.C, self.X, self.T)
        return N1

    def second_time_choice(self, time):
        N2 = BUCB(time, self.r2, self.C, self.X, self.T)
        return N2

    def third_time_choice(self, time):
        N3 = BUCB(time, self.r2, self.C, self.X, self.T)
        return N3

    def collision(self, channel):
        self.T[channel] += 1

    def first_time_reward(self, channel, row):
        reward = 0
        if row[channel] == "0":
            self.X[channel] += 1
            reward = 1
            self.total_reward += 1
            self.r1 = np.random.randint(1, self.U)
        self.T[channel] += 1
        return reward

    def second_time_reward(self, channel, row):
        reward = 0
        if row[channel] == "0":
            self.X[channel] += self.gamma
            reward = self.gamma
            self.total_reward += self.gamma
            self.r2 = np.random.randint(self.U + 1, min(2 * self.U, self.C))
        self.T[channel] += 1
        return reward

    def third_time_reward(self, channel, row):
        reward = 0
        if row[channel] == "0":
            self.X[channel] += self.epsilon
            reward = self.epsilon
            self.total_reward += self.epsilon
            self.r2 = np.random.randint(self.U + 1, min(2 * self.U, self.C))
        self.T[channel] += 1
        return reward
