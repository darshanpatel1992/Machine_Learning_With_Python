# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 00:51:54 2018

@author: darshan patel
"""
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import Dataset
dataset = pd.read_csv('Mall_Customers.csv')
X= dataset.iloc[:,[3,4]].values

#Use Dendrogram to find optimal number of clusters
#k-means Lboow method to find optimal number of clustering
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Ecludian Distances')
plt.show()
#We get to know that Dendrogram =5 clusters

#Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc= hc.fit_predict(X)

#Visullizing Clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='red',label='cluster1')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='blue',label='cluster1')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='green',label='cluster1')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='cyan',label='cluster1')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='magenta',label='cluster1')
plt.title('Clients of Cluster')
plt.xlabel('Annual Income')
plt.ylabel('Score')
plt.legend()
plt.show()









