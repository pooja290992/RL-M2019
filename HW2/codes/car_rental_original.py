
# coding: utf-8

# In[1]:


import matplotlib
from scipy.stats import poisson
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import trange 


# In[2]:


class CarRental:
    def __init__(self,const_ret_cars = True):
        self.max_cars = 20
        self.max_move_cars = 5
        self.rent_req_loc1 = 3
        self.rent_req_loc2 = 3
        self.ret_loc1 = 3
        self.ret_loc2 = 2
        self.discount = 0.9
        self.rent_credit = 10
        self.move_car_cost = 2
        self.actions = np.arange(-self.max_move_cars, self.max_move_cars + 1)
        self.truncate = 11
        self.values = np.zeros((self.max_cars + 1, self.max_cars + 1))
        self.policy = np.zeros(self.values.shape, dtype=np.int)
        self.const_ret_cars = const_ret_cars
        self.poisson_prob_dict = dict()


    def compPoissonProb(self,n, lam):
        key = n * 10 + lam
        if key not in self.poisson_prob_dict:
            self.poisson_prob_dict[key] = poisson.pmf(n, lam)
        return self.poisson_prob_dict[key]

    def getValues(self):
        return self.values

    def getPolicy(self):
        return self.policy

    def solveBellman(self,state, action, state_value):

        returns = 0.0

        returns -=  self.move_car_cost* abs(action)


        for req_loc1 in range(self.truncate):
            for req_loc2 in range(self.truncate):

                prob = self.compPoissonProb(req_loc1, self.rent_req_loc1) *                     self.compPoissonProb(req_loc2, self.rent_req_loc2)

                num_cars_loc1, num_cars_loc2 = min(state[0] - action, self.max_cars), min(state[1] + action, self.max_cars)

                rental_loc1, rental_loc2 = min(num_cars_loc1, req_loc1), min(num_cars_loc2, req_loc2)


                reward = (rental_loc1 + rental_loc2) * self.rent_credit
                
                num_cars_loc1 -= rental_loc1
                num_cars_loc2 -= rental_loc2

                if self.const_ret_cars:
                    ret_cars_loc1, ret_cars_loc2 = self.ret_loc1, self.ret_loc2
                    num_cars_loc1, num_cars_loc2 = min(num_cars_loc1 + ret_cars_loc1, self.max_cars), min(num_cars_loc2 + ret_cars_loc2, self.max_cars)
                    returns += prob * (reward + self.discount * state_value[num_cars_loc1, num_cars_loc2])
                else:
                    for ret_cars_loc1 in range(self.truncate):
                        for ret_cars_loc2 in range(self.truncate):
                            prob_return = self.compPoissonProb(
                                ret_cars_loc1, self.ret_loc1) * self.compPoissonProb(ret_cars_loc2, self.ret_loc2)
                            num_cars_loc1_ret, num_cars_loc2_ret = min(num_cars_loc1 + ret_cars_loc1, self.max_cars), min(num_cars_loc2 + ret_cars_loc2, self.max_cars)
                            prob_ret = prob_return * prob
                            returns += prob_ret * (reward + self.discount * state_value[num_cars_loc1_ret, num_cars_loc2_ret])
        return returns


    def policyEvaluation(self,value,policy):
        while True:
                old_value = value.copy()
                for i in range(self.max_cars + 1):
                    for j in range(self.max_cars + 1):
                        new_state_value = self.solveBellman([i, j], policy[i, j], value)
                        value[i, j] = new_state_value
                max_delta = abs(old_value - value).max()
                print(f'max delta : {max_delta}')
                if max_delta < 1e-4:
                    break

        return value 

    def policyImprovement(self,policy,value):
        policy_stable = True
        for i in range(self.max_cars + 1):
            for j in range(self.max_cars + 1):
                old_action = policy[i, j]
                action_returns = []
                for action in self.actions:
                    if (0 <= action <= i) or (-j <= action <= 0):
                        action_returns.append(self.solveBellman([i, j], action, value))
                    else:
                        action_returns.append(-np.inf)
                new_action = self.actions[np.argmax(action_returns)]
                policy[i, j] = new_action
                if policy_stable and old_action != new_action:
                    policy_stable = False
        print(f'policy stable {policy_stable}')
        return policy, policy_stable


    def figure_4_2(self):
        iter_cnt = 0
        
        _, axes = plt.subplots(2, 3, figsize=(40, 20))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        axes = axes.flatten()
        
        while True:
            fig = sns.heatmap(np.flipud(self.policy), cmap="YlGnBu", ax=axes[iter_cnt])
            fig.set_ylabel('# cars at first location', fontsize=30)
            fig.set_yticks(list(reversed(range(self.max_cars + 1))))
            fig.set_xlabel('# cars at second location', fontsize=30)
            fig.set_title('policy {}'.format(iter_cnt), fontsize=30)

            self.values = self.policyEvaluation(self.values,self.policy)

            # policy improvement
            self.policy, policy_stable = self.policyImprovement(self.policy,self.values)

            if policy_stable:
                fig = sns.heatmap(np.flipud(self.values), cmap="YlGnBu", ax=axes[-1])
                fig.set_ylabel('# cars at first location', fontsize=30)
                fig.set_yticks(list(reversed(range(self.max_cars + 1))))
                fig.set_xlabel('# cars at second location', fontsize=30)
                fig.set_title('optimal value', fontsize=30)
                break

            iter_cnt += 1
        print('iterations:',iter_cnt+1)
        #plt.savefig('../images/figure_4_2.png')
        #plt.close()
        plt.show()



# In[ ]:


car = CarRental()
car.figure_4_2()


# In[ ]:


car = CarRental(const_ret_cars = False)
car.figure_4_2()

