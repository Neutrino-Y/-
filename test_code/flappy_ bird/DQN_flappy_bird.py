"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/
Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from RL_flappy_bird import flappy_bird
import gym
import matplotlib.pyplot as plt
import time
import adabound

# Hyper Parameters
BATCH_SIZE = 64             # 每一次从记忆库中取出记忆样本的数量
LR = 0.005                   # learning rate
EPSILON = 0.99               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 500
env = flappy_bird()
N_ACTIONS = 2
N_STATES = 2
ENV_A_SHAPE = 0


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 100)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        # self.optimizer = adabound.AdaBound(self.eval_net.parameters(), lr=LR, final_lr=0.01)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net(x)
            action = torch.max(actions_value, 1)[1].data.numpy() # 【1】指向动作数据的指针
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        # 顺序覆盖
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict()) # eval_net每一次学习都会更新，target_net则每隔一段时间替换为eval_net
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate 与上一步不同的原因是要找出最大的可能奖励
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)【0】只索引奖励而不是动作
        loss = self.loss_func(q_eval, q_target)     # 可以同时将一个BATCH的差值用于更新网络

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 每一次训练后，现实网络都更适应当前批训练的数据，但如果每一次都与自己更新，效率更低，利用多次训练后的综合数据，将目标网络更新
dqn = DQN()

print('\nCollecting experience...')
draw_Data=[[],[]]

env.start()
plt.ion()

while True:
    for i_episode in range(500):
        s = [500,0]
        ep_r = 0
        env.reset()
        while True:
            # env.render()
            a = dqn.choose_action(s)

            # take action
            s_, done, score = env.step(a)

            # modify the reward
            if not done:
                r = 1
                try:
                    r += 0.1/(s_[1]-10)
                except:
                    pass
            else:
                r = -1000

            if s_[1] > 300:
                r = -1000


            dqn.store_transition(s, a, r, s_)
            # time.sleep(0.15)
            ep_r += r
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2))
                    draw_Data[0].append(i_episode)
                    draw_Data[1].append(score)

            if done:
                x = np.array(draw_Data[0])
                y = np.array(draw_Data[1])
                plt.plot(x, y)
                plt.pause(0.001)
                # plt.close()
                break
            s = s_
