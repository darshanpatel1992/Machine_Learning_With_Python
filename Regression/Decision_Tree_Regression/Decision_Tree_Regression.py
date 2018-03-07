# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 02:34:49 2018

@author: darshan patel
"""
#Decision Tree Regression
#Data Preprocessing
# impport the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X= dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Splitting the dataset into Training Set and Test Set
#from sklearn.cross_validation import train_test_split
#X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train=sc_X.fit_transform(X_train)
#X_test=sc_X.fit_transform(X_test)


#Fitting Decision Tree Regression model to dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state= 0)
regressor.fit(X,y)

#Predicting new Salaries with Decision_Tree regression model
y_pred=regressor.predict(6.5)

#Visulising Regression (For highr Resolution)
X_grid =np.arange(min(X),max(X),0.01)
X_grid= X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth of bluff(Decision Tree Regression Model)')
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()


