#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from sklearn.metrics import accuracy_score 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# In[2]:


# create a dataframe with two columns "Rate" and "Comment" to make the preprocessing easy
reviewTrain = pd.read_csv("train_file.txt", sep='\t',header=None,names=['Rate','Comment'], skip_blank_lines=True, dtype = str, na_filter=False,infer_datetime_format=True, error_bad_lines=False)


# In[3]:


reviewTrain.head()


# In[4]:


reviewTest = pd.read_csv("test_file.txt", sep='\t',header=None,names=['Comment'], skip_blank_lines=True, dtype = str, na_filter=False,infer_datetime_format=True, error_bad_lines=False)


# In[5]:


"""
Function: preProcessing(corpus): 
It does many necessary preprocessing to data as follow:
1. delete anything that does not belong to [a-z] or [A-Z] from the data
2. transfer all characters to lower case
3. remove all stop words from the data except the word "not" because it is necessary information
4. apply stemming to data

Parameter: dataframe of reviewDataset 
return: list of preprocessed corpus of reviewDataset according to the above 4 steps
"""
def preProcessing(reviewDataset):
        preprocessedCorpus = []
        # iterate over ratring in datafame
        for i in range(len(reviewDataset)):
            #delete all symbols except a-z and A-Z in the Comment column
            comment = re.sub('[^a-zA-Z]', ' ', reviewDataset['Comment'][i])
            comment = comment.lower()
            comment = comment.split()
            porterstemmer = PorterStemmer() 
            sw = stopwords.words('english')
            #exclude not from stop word set
            sw.remove('not')
            # apply stemming to words in comment 
            comment = [porterstemmer.stem(word) for word in comment if not word in set(sw)]
            #join the words 
            comment = ' '.join(comment)
            #add train data to corpus
            preprocessedCorpus.append(comment)
        return preprocessedCorpus


# In[6]:


#preprocess data
sentiment = reviewTrain.iloc[:,0].values
preprocessedReviewTrain = preProcessing(reviewTrain)
preprocessedReviewTest = preProcessing(reviewTest)


# In[7]:


# create spase matrix using Tf-idf vectorizer for training and testing data
v = TfidfVectorizer(ngram_range=(2,5), max_features=30000)
tfidfTrain = v.fit_transform(preprocessedReviewTrain)
tfidfTest = v.transform(preprocessedReviewTest)


# In[8]:


def knn(k, testTfidf, trainTfidf, x_test, y_train):
    prediction = []    
    for i, l in enumerate(x_test):
        # calculate cosine similarity with the help of cosine_similarity in sklearn 
        cs = cosine_similarity(tfidfTest[i], tfidfTrain).flatten()
        # utilizing numpy.argsort to find the K neighbors indices      
        nearestNeighborIndices = cs.argsort()[:-k:-1]
        #get the list of k nearest neighbor  from the sentiment  
        nearestNeighborList = y_train[nearestNeighborIndices]
        
        # convert nearestNeighborList from string type to int type
        integer_map=map(int, nearestNeighborList)
        nearestNeighborList = list(integer_map)
        
        ''' if sum of the nearestNeighborList is >0, then this mean that the +1 class is the majority 
        if the sum is zero, then the positive and negative are equal, but the program will classify it as positive
        if the the sum is negative, then the majority of the neighbors are "-1" 
        '''
        decision = sum(nearestNeighborList)
        
        if decision >= 0:
          prediction.append("+1")
        else:
          prediction.append("-1")
            
    return pd.DataFrame(prediction)


# In[11]:


submission=knn(236,tfidfTest, tfidfTrain, preprocessedReviewTest, sentiment) 


# In[12]:


np.savetxt('./submission.txt', submission,fmt='%s')


# In[ ]:




