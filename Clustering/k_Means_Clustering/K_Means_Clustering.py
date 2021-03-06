# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 06:23:21 2018

@author: darshan patel
"""
#K-Means Clustering
#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values


#Using Lbow method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range (1,11):
    kmeans =KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('no of Clusters')
plt.ylabel('WCSS')
plt.show()

#Applying Kmeans to the dataset
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)

#Visullizing the clusters
plot=(plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='cluster1'),
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='cluster1'),
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='cluster1'),
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='cluster1'),
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='cluster1'),
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroid'),
plt.title('Clients of Cluster'),
plt.xlabel('Annual Income'),
plt.ylabel('Score'),
plt.legend(),
plt.show())
















    
    
