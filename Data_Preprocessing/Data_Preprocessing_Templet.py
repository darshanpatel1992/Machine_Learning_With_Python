# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 04:30:50 2018

@author: darshan patel
"""

#Data Preprocessing
# impport the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv("Data.csv")
X= dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

#Splitting the dataset into Training Set and Test Set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train=sc_X.fit_transform(X_train)
#X_test=sc_X.fit_transform(X_test)
#
