import csv
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import numpy as np

# Loss
loss = pd.read_csv('C:\\Users\\Darryl\\PycharmProjects\\BEP_Snake\\loss.csv')

loss_plot = loss.plot('Epoch', 'Mean Loss')
loss_plot.set_xticks(loss['Epoch'])
plt.show()

# Reward
rewards = pd.read_csv('C:\\Users\\Darryl\\PycharmProjects\\BEP_Snake\\rewards.csv')

rewards_plot = rewards.plot('Epoch', 'Mean Reward')
rewards_plot.set_xticks(rewards['Epoch'])
plt.show()

# Q values
q_values = pd.read_csv('C:\\Users\\Darryl\\PycharmProjects\\BEP_Snake\\q.csv')

q_plot = q_values.plot('Epoch', 'Mean Q')
q_plot.set_xticks(q_values['Epoch'])
plt.show()

# Time
data = pd.read_csv('C:\\Users\\Darryl\\PycharmProjects\\BEP_Snake\\data_mines.csv', names=['scores', 'mean_score', 'times', 'mean_time'])

data['game'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26, 27, 28,
                29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]

plt.plot('game', 'scores', data=data)
plt.plot('game', 'mean_score', data=data)
plt.xlabel("Number of Games")
plt.ylabel("Score")
plt.title("Moving Target Snake DQN Scores on Moving Target Game")
plt.show()

data_classic = pd.read_csv('C:\\Users\\Darryl\\PycharmProjects\\BEP_Snake\\data.csv', names=['scores', 'mean_score', 'times', 'mean_time'])
data_mt = pd.read_csv('C:\\Users\\Darryl\\PycharmProjects\\BEP_Snake\\data_movingtarget.csv', names=['scores', 'mean_score', 'times', 'mean_time'])

scores_c = data_classic['scores']
scores_mt = data_mt['scores']
