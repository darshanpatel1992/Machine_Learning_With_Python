# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 01:45:04 2018

@author: darshan patel
"""
#Support vector Regression.
#Data Preprocessing
# impport the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X= dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

#Fitting SVR Regression model to dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

#Predicting new Salaries with SVR regression model
y_pred=regressor.predict(6.5)

#Visulising SVR Regression
plt.scatter(X,y,color = 'red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth of bluff( Regression Model)')
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()

#Now we can also apply feature scalling object to the 

#import the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X= dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)

#Fitting SVR Regression model to dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

#Predicting new Salaries with SVR regression model
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visulising SVR Regression
plt.scatter(X,y,color = 'red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth of bluff( Regression Model)')
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()

#Visulising SVR Regression (For highr Resolution)
X_grid = np.array(min(X),max(X),0.1)
X_grid= X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color = 'red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth of bluff( Regression Model)')
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()


