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
from Gym_CarRacing_v02 import CarRacing
import time
import adabound
import os

# Hyper Parameters
BATCH_SIZE = 128
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 200   # target update frequency
MEMORY_CAPACITY = 5000
N_STATES = 6
N_ACTIONS = 4
ENV_A_SHAPE = 0

TRACK_WEIGHT = 1.0
SPEED_WEIGHT = 0.7
PART_WEIGHT = 0.1
MIDDLE_WEIGHT = 18.5


env = CarRacing()


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
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
        # self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.optimizer = adabound.AdaBound(self.eval_net.parameters(), lr=LR, final_lr=0.01)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
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
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()
evolution_circel = 0
evolution_score =[]
print('\nCollecting experience...')
first_circel_over = False


for i_episode in range(4000):

    if dqn.memory_counter > MEMORY_CAPACITY:

        if evolution_circel ==5:

            parent_index = sorted(evolution_score, key = lambda score: score[1], reverse=True)[0][0]
            print("choose["+str(parent_index)+"]  Score="+str(sorted(evolution_score, key = lambda score: score[1], reverse=True)[0][1]))
            os.rename("./save/"+str(parent_index)+'_eval_net_params.pkl', "./save/parent_eval_net_params.pkl")
            os.rename("./save/" + str(parent_index) + '_target_net_params.pkl', "./save/parent_target_net_params.pkl")
            os.rename("./save/" + str(parent_index) + '_dqn.memory.npy', "./save/parent_dqn.memory.npy")

            last_best_score = sorted(evolution_score, key = lambda score: score[1], reverse=True)[0][1]
            evolution_score = []
            evolution_score.append([0,last_best_score])
            evolution_circel = 1

            dqn.eval_net.load_state_dict(torch.load("./save/parent_eval_net_params.pkl"))
            dqn.target_net.load_state_dict(torch.load("./save/parent_target_net_params.pkl"))
            dqn.memory = np.load("./save/parent_dqn.memory.npy")

            torch.save(dqn.eval_net.state_dict(), "./save/0_eval_net_params.pkl")
            torch.save(dqn.target_net.state_dict(), "./save/0_target_net_params.pkl")
            np.save("./save/0_dqn.memory", dqn.memory)

            first_circel_over = True

        if first_circel_over:

            dqn.eval_net.load_state_dict(torch.load("./save/parent_eval_net_params.pkl"))
            dqn.target_net.load_state_dict(torch.load("./save/parent_target_net_params.pkl"))
            dqn.memory = np.load("./save/parent_dqn.memory.npy")



    s = env.reset()
    ep_r = 0
    first_front = 10

    for i in range(50):
        env.step([0.0,0.0,0.0])
        env.render()
    while True:
        while first_front > 0:
            env.step([0.0, 1.0, 0.0])
            env.render()
            first_front -= 1
        env.render()
        a = dqn.choose_action(s)
        actions = [0.0,0.0,0.0]

        if a == 0:  actions[0] = -0.5
        if a == 1:  actions[0] = +0.5
        if a == 2:  actions[1] = +1.0
        if a == 3:  actions[2] = +0.8  # set 1.0 for wheels to block to zero rotation

        # take action
        s_, r, done, info= env.step(actions)
        speed = s_[5]

        s1 = (-0.5*s_[0]**2+20*s_[0])*1/400 - 0.2
        s2 = (-0.5*s_[4]**2+20*s_[4])*1/400 - 0.2
        r1 =  round((s1+s2)/2,2)

        r2 = round(speed**0.5*SPEED_WEIGHT*0.3,2)
        r = round(r1+r2+r*PART_WEIGHT,2)*0.5

        # modify the reward

        if speed == 0:
            r = -0.005
        if s_[0] <= 1 or s_[4] <= 1:
            r = -0.02
            done = True
        if r2<0.2 :
            r = -0.01
        dqn.store_transition(s, a, r, s_)
        #print("memory_counter:",dqn.memory_counter )

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            if dqn.memory_counter%10 == 0:
                dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

                evolution_score.append([evolution_circel])
                evolution_score[evolution_circel].append(round(ep_r, 2))

                print("Save " + str(evolution_circel) + ":" + str(ep_r))
                print(evolution_score)

                torch.save(dqn.eval_net.state_dict(), "./save/" + str(evolution_circel) + '_eval_net_params.pkl')
                torch.save(dqn.target_net.state_dict(), "./save/" + str(evolution_circel) + '_target_net_params.pkl')
                np.save("./save/" + str(evolution_circel) + '_dqn.memory', dqn.memory)

                evolution_circel += 1

        if done:
            break
        s = s_