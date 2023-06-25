import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import deque
from game import SnakeAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import sys
import time
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

MAX_MEMORY = 50_000
BATCH_SIZE = 1000
LR = 0.01

train_test = 'train'
model = Linear_QNet(33, 128, 3)
if train_test == 'test':
    PATH = 'C:\\Users\\Darryl\\PycharmProjects\\BEP_Snake\\model\\final_movingtarget.pth'
    model.load_state_dict(torch.load(PATH))
    model.eval()


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.95
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = model
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):

        # unused
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.L
        dir_r = game.direction == Direction.R
        dir_u = game.direction == Direction.U
        dir_d = game.direction == Direction.D

        state = [
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
            ]

        game.get_surface_().flatten()
        surface = game.get_surface_()

        return surface, state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 100 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 100) < self.epsilon and train_test == 'train':
            move = random.randint(0, 2)
            final_move[move] = 1
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)

        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move, prediction

def play():
    iterator = 0
    epoch = 1

    plot_scores = []
    plot_mean_scores = []
    total_score = 0

    plot_loss = []

    plot_reward = [['Epoch', 'Mean Reward']]
    rewards = []

    plot_q = [['Epoch', 'Mean Q']]
    qs = []

    times = []
    avg_time = []
    total_time = 0

    record = 0
    agent = Agent()
    game = SnakeAI()

    while True:
        start_time = time.time()

        state_old = agent.get_state(game)[0]

        final_move = agent.get_action(state_old)[0]

        q = max(agent.get_action(state_old)[1]).item()
        qs.append(q**2)

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)[0]

        rewards.append(reward)

        if train_test == 'train':
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            end_time = time.time()

            game.reset()
            agent.n_games += 1

            if train_test == 'train':
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()

            if agent.n_games == 1000:
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            time_taken = end_time-start_time
            times.append(time_taken)
            total_time += time_taken

            if 100 < agent.n_games <= 200:
                plot_loss.extend(agent.trainer.ep_loss)
                agent.trainer.ep_loss.clear()

            mean_time = total_time / agent.n_games
            avg_time.append(mean_time)

            if agent.n_games == 100:
                with open('data_mines.csv', 'w') as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(plot_scores, plot_mean_scores, times, avg_time))

            if agent.n_games > 200:
                f_loss = open('C:\\Users\\Darryl\\PycharmProjects\\BEP_Snake\\loss.csv', 'w+')
                writer_loss = csv.writer(f_loss)
                writer_loss.writerows(plot_loss)
                f_loss.close()

            if agent.n_games > 100:
                if iterator % 5 == 0:
                    #plot_loss.append([epoch, sum(agent.trainer.ep_loss)/len(agent.trainer.ep_loss)])
                    agent.trainer.ep_loss.clear()

                    plot_reward.append([epoch, sum(rewards)/len(rewards)])
                    rewards.clear()

                    plot_q.append([epoch, sum(qs)/len(qs)])
                    qs.clear()
                    epoch += 1

                if iterator == 100:
                    f_loss = open('C:\\Users\\Darryl\\PycharmProjects\\BEP_Snake\\loss.csv', 'w')
                    writer_loss = csv.writer(f_loss)
                    writer_loss.writerows(plot_loss)
                    f_loss.close()

                    f_rewards = open('C:\\Users\\Darryl\\PycharmProjects\\BEP_Snake\\rewards.csv', 'w')
                    writer_rewards = csv.writer(f_rewards)
                    writer_rewards.writerows(plot_reward)
                    f_rewards.close()

                    f_qs = open('C:\\Users\\Darryl\\PycharmProjects\\BEP_Snake\\q.csv', 'w')
                    writer_q = csv.writer(f_qs)
                    writer_q.writerows(plot_q)
                    f_qs.close()

                iterator += 1

if __name__ == '__main__':
    play()