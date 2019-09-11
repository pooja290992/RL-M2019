
# coding: utf-8

# In[1]:


import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math 
from tqdm import trange
from matplotlib.table import Table
import time


# In[30]:


class GridWorld:
    def __init__(self,grid_size = 5, gamma = 0.9, penalty = -1.0, theta = 1e-3):
        self.grid_size = grid_size
        self.discount = gamma
        self.actions = [np.array([0, -1]),
                   np.array([-1, 0]),
                   np.array([0, 1]),
                   np.array([1, 0])]
        self.action_prob = 0.25#1/len(self.actions)
        self.theta = theta
        print('action prob : ',self.action_prob)
        self.penalty_reward = penalty
        self.re_init()
    
    def re_init(self):
        self.values = np.zeros((self.grid_size, self.grid_size))
        #self.policy = np.zeros((self.grid_size, self.grid_size, len(self.actions)), dtype=np.int)
        #self.policy = np.zeros(self.values.shape, dtype=np.int)
        self.policy = np.ones((self.grid_size, self.grid_size, len(self.actions)))/len(self.actions)
        #self.policy = np.random.randint(0,len(self.actions), size=(self.grid_size , self.grid_size ))
        #print(self.policy)
        
    
    def checkTerminal(self,state):
        x, y = state
        if x == 0 and y == 0:
            return 1
        elif (x == self.grid_size - 1 and y == self.grid_size - 1):
            return 1
        else : 
            return 0
        
    
    def step(self, state, action):
        #print(state)
        if self.checkTerminal(state):
            next_state = state
            reward = 0
        else:
            next_state = (np.array(state) + action).tolist()
            x, y = next_state
            if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
                next_state = state
            reward = self.penalty_reward     
            
        return next_state, reward
    
    
    def compStateValue(self, in_place_flag = True, random_flag = True):
        new_state_values = np.zeros((self.grid_size, self.grid_size))
        iter_cnt = 0
        while True:
            if in_place_flag:
                state_values = new_state_values
            else:
                state_values = new_state_values.copy()
            old_state_values = state_values.copy()

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    value = 0
                    for action in self.actions:
                        (next_i, next_j), reward = self.step([i, j], action)
                        value += self.action_prob * (reward + self.discount * state_values[next_i, next_j])
                    new_state_values[i, j] = value

            max_delta_value = abs(old_state_values - new_state_values).max()
            iter_cnt += 1
            if max_delta_value < 1e-3:
                break

        return new_state_values, iter_cnt
    
    
    def compValueIteration(self):
        new_state_values = np.zeros((self.grid_size, self.grid_size))
        policy = np.zeros((self.grid_size, self.grid_size))
        iter_cnt = 0
        log_dict = {}
        while True:
            #delta = 0
            log_dict[iter_cnt] = {}
            state_values = new_state_values.copy()
            old_state_values = state_values.copy()
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    values = []
                    for action in self.actions:
                        (next_i, next_j), reward = self.step([i, j], action)
                        values.append(reward + self.discount*state_values[next_i, next_j])
                    new_state_values[i, j] = np.max(values)
                    policy[i, j] = np.argmax(values)
                    #delta = max(delta, np.abs(old_state_values[i, j] - new_state_values[i, j]))
            delta = np.abs(old_state_values - new_state_values).max()        
            print(f'Difference: {delta}')
            print(new_state_values)
            log_dict[iter_cnt]['VI'] = new_state_values.copy()
            #print(log_dict[iter_cnt]['VI'])
            iter_cnt += 1
            if delta < self.theta:
                break
            
            #print(new_state_values)
        #print(log_dict)
        return new_state_values, policy, iter_cnt, log_dict
     
    
    
    def policyEvaluation(self,policy,new_state_values):
        #new_state_values = np.zeros((self.grid_size, self.grid_size))
        iter_cnt = 0
        while True:
            delta = 0 
            state_values = new_state_values.copy()
            old_state_values = state_values.copy()
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    value = 0
                    for action,action_prob in enumerate(policy[i, j]):
                        (next_i, next_j), reward = self.step([i, j], self.actions[action])
                        value += action_prob * (reward + self.discount * state_values[next_i, next_j])
                    new_state_values[i, j] = value
                    delta = max(delta, np.abs(old_state_values[i, j] - new_state_values[i, j]))
            iter_cnt += 1
            #print(f'Difference: {delta}')
            if delta < self.theta:
                break
            
            #print(new_state_values)
        return new_state_values
    
    
    def policyImprovement(self, policy, values, actions):
        policy_stable = True
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                #old_action = policy[i, j]
                old_action = np.argmax(policy[i, j])
                act_cnt = 0
                expected_rewards = []
                for action in self.actions:
                    (next_i, next_j), reward = self.step([i, j], action)
                    expected_rewards.append(self.action_prob * (reward + self.discount * values[next_i, next_j]))
                new_action = np.argmax(expected_rewards)
                #print('new_action : ',new_action)
                #print('old_action : ',old_action)
                if old_action != new_action:
                    policy_stable = False
                #policy[i, j] = new_action
                policy[i, j] = np.eye(len(self.actions))[new_action]
        return policy, policy_stable
        
        
        
    def solve(self):
        iterations = 0
        total_start_time = time.time()
        log_dict = {}
        
        while True:
            log_dict[iterations] = {}
            start_time = time.time()
            self.values = self.policyEvaluation(self.policy,self.values)
            elapsed_time = time.time() - start_time
            print(f'PE => Elapsed time {elapsed_time} seconds')
            start_time = time.time()
            log_dict[iterations]['PE'] = self.values    
            self.policy, policy_stable = self.policyImprovement(self.policy,self.values, self.actions)
            r,c,v = self.policy.shape
            policy = np.zeros((self.grid_size,self.grid_size))
            for i in range(r):
                for j in range(c):
                    policy[i, j] = np.argmax(self.policy[i,j])
            log_dict[iterations]['PI'] = policy
            log_dict[iterations]['Pol_stable'] = policy_stable
            elapsed_time = time.time() - start_time
            print(f'PI => Elapsed time {elapsed_time} seconds')
            
            iterations += 1
            if policy_stable:
                break
                
        total_elapsed_time = time.time() - total_start_time
        print(f'Optimal policy is reached after {iterations} iterations in {total_elapsed_time} seconds')
        #print(self.policy)  
                    
        return self.values, policy, iterations, log_dict


# In[31]:


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
    


# In[50]:


def figure_4_1():
    grid = GridWorld(grid_size = 4, gamma = 1.0)
    values, sync_iteration = grid.compStateValue(in_place_flag = False)
    title = 'Grid world'
    plotGrid(values,title)
    print('Synchronous: {} iterations'.format(sync_iteration))
    print('*'*100)
    print('*'*100)
    values, policy, iter_cnt,log_dict1 = grid.compValueIteration()
    #print(log_dict1)
    title = 'Grid world : V(s) with Value Iteration'
    plotGrid(values,title)
    title = 'Grid world : pi(s) with Value Iteration'
    plotGrid(policy,title)
    print('iterations : {} '.format(iter_cnt))
    print('Synchronous: {} iterations'.format(sync_iteration))
    grid2 = GridWorld(grid_size = 4, gamma = 1.0)
    values, policy, iter_cnt, log_dict2 = grid2.solve()
    title = 'Grid world : V(s) with Policy Iteration'
    plotGrid(values,title)
    title = 'Grid world : pi(s) with Policy Iteration'
    plotGrid(policy,title)
    #print(log_dict1)
    #print(log_dict2)
    return log_dict1, log_dict2


# In[51]:


log_dict1,log_dict2 = figure_4_1()
#print(log_dict1)


# In[52]:


log_df1 = pd.DataFrame.from_dict(log_dict1,orient='index').reset_index()
log_df2 = pd.DataFrame.from_dict(log_dict2,orient='index').reset_index()


# In[53]:


#figure_4_1()
display(log_df1)
display(log_df2)


# In[54]:


path_vi = '../Results/value_iteration_log2.csv'
path_pi = '../Results/policy_iteration_log2.csv'
log_df1.to_csv(path_vi,sep = ',')
log_df2.to_csv(path_pi,sep = ',')

