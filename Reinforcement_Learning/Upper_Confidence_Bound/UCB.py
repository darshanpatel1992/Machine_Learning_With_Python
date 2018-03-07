# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 03:03:36 2018

@author: darshan patel
"""
#UCB Algorithm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

#Implimenting UCB algorithms:
import math
N=10000
d=10
ads_selected=[]
number_of_selections=[0]*d
sums_of_rewards=[0]*d
total_rewards=0
for n in range(0,N):
    ad=0
    max_upper_bound=0
    for i in range(0,d):
        if(number_of_selections[i]>0):
            average_rewards=sums_of_rewards[i]/number_of_selections[i]
            delta_i=math.sqrt(3/2*math.log(n+1)/number_of_selections[i])
            upper_bound=average_rewards+delta_i
        else:
            upper_bound=1e400
        if upper_bound > max_upper_bound:
            max_upper_bound=upper_bound
            ad=i
    ads_selected.append(ad)
    number_of_selections[ad]= number_of_selections[ad]+1
    rewards= dataset.values[n,ad]
    sums_of_rewards[ad]=sums_of_rewards[ad]+rewards
    total_rewards=total_rewards+rewards

#Visullizing the result
plt.hist(ads_selected)           
plt.title('Histogram of selections')
plt.xlabel('Ads')
plt.ylabel('no of time each Ad was selected')
plt.show()
             
        






















