import csv
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from tqdm import tqdm

from cogn_user import CognitiveUser
from cogn_bucb import BUCB


def run_algorithm(C, U, H, gamma, epsilon, file_, three_stage=False):
    users = [CognitiveUser(C, U, gamma, epsilon) for _ in range(U)]
    rows = []
    with open(file_, "r") as file:
        rows = list(csv.reader(file))

    total_reward = 0
    total_rewards = []

    for t in tqdm(range(C)):
        d1 = {}
        d2 = {}
        d3 = {}
        users_that_didnt_make_it_through = []
        users_that_didnt_make_it_through_again = []
        reward = 0
        # print()
        for user in users:
            channel = user.random_array[t]
            # print(channel, end=", ")
            if channel in d1:
                user.collision(channel)
                user2 = d1[channel][1]
                users_that_didnt_make_it_through.append(user)
                users_that_didnt_make_it_through.append(user2)
                # d1[channel] += (1, user)
            else:
                d1[channel] = (1, user)
                reward += float(user.first_time_reward(channel, rows[t]))

        total_reward += reward
        # total_rewards.append(total_reward)
    for t in tqdm(range(C, H)):
        reward = 0
        d1 = {}
        d2 = {}
        d3 = {}
        users_that_didnt_make_it_through = []
        users_that_didnt_make_it_through_again = []
        # print()
        for user in users:
            # print(channel, end=", ")
            channel = user.first_time_choice(t)
            if channel in d1:
                user.collision(channel)
                user2 = d1[channel][1]
                users_that_didnt_make_it_through.append(user)
                users_that_didnt_make_it_through.append(user)
            else:
                d1[channel] = (1, user)
                reward += float(user.first_time_reward(channel, rows[t]))
        # print()
        for user in users_that_didnt_make_it_through:
            # print(channel, end=", ")
            channel = user.second_time_choice(t)
            if channel in d2:
                user.collision(channel)
                if not three_stage:
                    reward = 0.0
                else:
                    user2 = d2[channel][1]
                    users_that_didnt_make_it_through_again.append(user)
                    users_that_didnt_make_it_through_again.append(user2)
            else:
                d2[channel] = (1, user)
                reward += float(user.second_time_reward(channel, rows[t]))

        if three_stage:
            for user in users_that_didnt_make_it_through_again:
                # print(channel, end=", ")
                channel = user.third_time_choice(t)
                if channel in d3:
                    user.collision(channel)
                    reward = 0.0
                else:
                    d3[channel] = (1, user)
                    reward += float(user.third_time_reward(channel, rows[t]))

        total_reward += reward
        total_rewards.append((t, total_reward / (U * t + 1)))

    print(total_reward / (U * t + 1))
    plt.plot(*zip(*total_rewards), label=f"three_stage={three_stage}")
    plt.title(f"C={C}, U={U}, H={H}, gamma={gamma}, epsilon={epsilon}, file={file_}")
    plt.legend()
    if not three_stage:
        run_algorithm(C, U, H, gamma, epsilon, file_, three_stage=True)
    plt.show()


if __name__ == "__main__":
    np.random.seed(30)
    run_algorithm(
                C=8, U=2, H=2000, gamma=0.85, epsilon=0.7, file_=f"./data/cogn-input-case1.csv"
    )
