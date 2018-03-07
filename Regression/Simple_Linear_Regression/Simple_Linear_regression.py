# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 04:38:14 2018

@author: darshan patel
"""

#Data Preprocessing
# impport the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv("Salary_Data.csv")
X= dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values

#Splitting the dataset into Training Set and Test Set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=1/3,
                                                random_state=0)

#Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train=sc_X.fit_transform(X_train)
#X_test=sc_X.fit_transform(X_test)
#


#Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predict Test Set Result
y_pred = regressor.predict(X_test)

#Visulising the Training Set result
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title("Salary vs Exp..(TrainingSet)")
plt.xlabel("Years of Exp")
plt.ylabel("Salary")
plt.show()

#Visulising the Test Set result
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title("Salary vs Exp..(Test Set)")
plt.xlabel("Years of Exp")
plt.ylabel("Salary")
plt.show()































