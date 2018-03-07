# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 05:21:35 2018

@author: darshan patel
"""
#Multiple_Linear_regression
#Data Preprocessing
# impport the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv("50_Startups.csv")
X= dataset.iloc[:,:-1].values
y = dataset.iloc[:, 4].values

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X= onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy variable trap
X=X[:,1:]

#Splitting the dataset into Training Set and Test Set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train=sc_X.fit_transform(X_train)
#X_test=sc_X.fit_transform(X_test)
#

#Fitting Multiple Linear regression model to Training set.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Prediciton of test Set Result
y_pred = regressor.predict(X_test)
y_pred

#Bulding optimal model using Backward Elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int), values=X,axis=1)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS= sm.OLS(endog=y,exog= X_opt).fit()
regressor_OLS.summary()

#iteration 2 remove x(3)
X_opt = X[:,[0,1,2,4,5]]
regressor_OLS= sm.OLS(endog=y,exog= X_opt).fit()
regressor_OLS.summary()


X_opt = X[:,[0,1,4,5]]
regressor_OLS= sm.OLS(endog=y,exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,4]]
regressor_OLS= sm.OLS(endog=y,exog= X_opt).fit()
regressor_OLS.summary()

#Building Backward Elimination with loop
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
X_opt






