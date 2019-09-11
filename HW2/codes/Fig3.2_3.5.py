
# coding: utf-8

# In[3]:


import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math 
from tqdm import trange
from matplotlib.table import Table


# In[58]:


class GridWorld:
    def __init__(self,grid_size = 5,a_xy = [0,1], a_prime_xy = [4, 1], b_xy = [0, 3], 
                 b_prime_xy = [2, 3], gamma = 0.9, a_reward = 10, b_reward = 5, penalty = -1.0):
        self.grid_size = grid_size
        self.A_xy  = a_xy
        self.A_prime_xy = a_prime_xy
        self.A_reward = a_reward
        self.B_xy = b_xy
        self.B_prime_xy = b_prime_xy
        self.B_reward = b_reward
        self.discount = gamma
        self.actions = [np.array([0, -1]),
                   np.array([-1, 0]),
                   np.array([0, 1]),
                   np.array([1, 0])]
        self.action_prob = 1/len(self.actions)
        print('action prob : ',self.action_prob)
        self.penalty_reward = penalty
        
    
    def step(self, state, action):
        if state == self.A_xy:
            return self.A_prime_xy, self.A_reward
        if state == self.B_xy:
            return self.B_prime_xy, self.B_reward
        next_state = (np.array(state) + action).tolist()
        x, y = next_state
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            reward = self.penalty_reward
            next_state = state
        else:
            reward = 0
        return next_state, reward
    
    
    def play(self,random_flag = True):
        value = np.zeros((self.grid_size, self.grid_size))
        cnt = 0
        while True:
            new_value = np.zeros_like(value)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    values = []
                    for action in self.actions:
                        (next_i, next_j), reward = self.step([i, j], action)
                        # bellman equation
                        if random_flag:
                            new_value[i, j] += self.action_prob * (reward + self.discount * value[next_i, next_j])
                        else:
                            values.append(reward + self.discount * value[next_i, next_j])
                    if not random_flag:
                        new_value[i, j] = np.max(values)    
#             print('value')
#             print(value)
#             print('new_value')
#             print(new_value)
            if np.sum(np.abs(value - new_value)) < 1e-3:
                return new_value
#             else:
#                 cnt += 1
#                 print('not converged')
#                 print('cnt : ',cnt)
            value = new_value


# In[67]:


def plotGrid(value,title):
    value = np.round(value, decimals=1)
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = value.shape
    width, height = 1.0 / ncols, 1.0 / nrows
    for (i, j), val in np.ndenumerate(value):
        tb.add_cell(i, j, width, height, text=val,
                    loc='center')
    ax.add_table(tb)
    plt.title(title)
    plt.show()
    


# In[68]:


def figure3_2():
    grid = GridWorld()
    value = grid.play()
    title = 'v(s) with random policy'
    plotGrid(value,title)


# In[73]:


def figure3_5():
    grid = GridWorld()
    value = grid.play(random_flag = False)
    title = r'$v_*(s)$' + ' with optimal policy'
    plotGrid(value,title)


# In[74]:


figure3_2()
figure3_5()

