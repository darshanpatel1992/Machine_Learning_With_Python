# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 04:06:54 2018

@author: darshan patel
"""

#Thompson Sampling  Algorithm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implimenting UCB algorithms:
import random
N=10000
d=10
ads_selected=[]
number_of_rewards_1=[0]*d
number_of_rewards_0=[0]*d
total_rewards=0
for n in range(0,N):
    ad=0
    max_random=0
    for i in range(0,d):
        random_beta=random.betavariate(number_of_rewards_1[i]+1,number_of_rewards_0[i]+1)
        if random_beta>max_random:
            max_random=random_beta
            ad=i    
    ads_selected.append(ad)
    rewards=dataset.values[n,ad]
    if rewards==1:
        number_of_rewards_1[ad]=number_of_rewards_1[ad]+1
    else:
        number_of_rewards_0[ad]=number_of_rewards_0[ad]+1
    total_rewards=total_rewards+rewards

#Visullizing the result
plt.hist(ads_selected)           
plt.title('Histogram of selections')
plt.xlabel('Ads')
plt.ylabel('no of time each Ad was selected')
plt.show()
             
        






















