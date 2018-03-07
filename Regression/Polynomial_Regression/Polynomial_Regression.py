# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 03:41:24 2018

@author: darshan patel
"""

#Polynomial Regression
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

#Fitting Linear Regression model to the dataset
from sklearn.linear_model import LinearRegression
linear_reg= LinearRegression()
linear_reg.fit(X,y)

#Fitting polynomial Regression model to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly= poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visulising Linear Regression
plt.scatter(X,y,color = 'red')
plt.plot(X,linear_reg.predict(X),color='blue')
plt.title('Truth of bluff(Linear Regression)')
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()
#Visulising polynomial Regression
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Truth of bluff(polynomial Regression)')
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()

#Predicting new Salaries with Linear regression model
linear_reg.predict(6.5)

#Predicting new Salaries with Polynomial regression model
lin_reg_2.predict(poly_reg.fit_transform(6.5))









