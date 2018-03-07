# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 04:56:16 2018

@author: darshan patel
"""
#Natural_Language_Processing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset(seprate by tab ,and ignore ("") in dataset)
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#Cleaning the text step1:clean . and replace by space
import re
review= re.sub('[^a-zA-Z]',' ',dataset['Review'][0])

#step2:All char by small
review=review.lower()
#step3:Remove words which not required
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
review= review.split() #split review in diffrent words
review = [word for word in review if not word in set(stopwords.words('english'))]
#step 4:Stamming of words like loved to love
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
review=([ps.stem(word) for word in review if not word in set(stopwords.words('english'))])

#step 5 
review = ' '.join(review)
    
#############################################################
#Cleaning the text for all 1000 rows
#################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset(seprate by tab ,and ignore ("") in dataset)
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus= []
for i in range(0,1000):
    review= re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review= review.split() #split review in diffrent words
    review = [word for word in review if not word in set(stopwords.words('english'))]
    ps=PorterStemmer()
    review=([ps.stem(word) for word in review if not word in set(stopwords.words('english'))])
    review = ' '.join(review)
    corpus.append(review)
########################################################################
#Create Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)    
X= cv.fit_transform(corpus).toarray()
y= dataset.iloc[:,1].values
#######################################################################3
#Classification model Naive Bayes
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)






































