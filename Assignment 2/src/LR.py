#!/usr/bin/env python
# coding: utf-8

# In[189]:


import time
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score 

import scipy.sparse as sp


# In[190]:


train_data = pd.read_table("train.txt", header=None, skip_blank_lines=False)


# In[191]:


test_data = pd.read_table("test.txt", header=None, skip_blank_lines=False)


# In[192]:


train_data.iloc[0, :]


# In[ ]:





# In[193]:


y=np.array(train_data[0])


# In[194]:


y


# In[195]:


x=train_data[1]


# In[196]:


x


# In[197]:


test=test_data[0]
test


# In[198]:


# #check for NaN values
# x.isnull().values.any()
# x.isnull().sum().sum()


# In[199]:


# test.isnull().values.any()
# test.isnull().sum().sum()


# In[200]:


#Ulilizing coo_matrix((entries, (r, c)), [shape=(M, N)]) to convert Training set matrix to a sparse matrix
def transform_to_sparse(instances):
    entries = []
    row = []
    col = []
    for i, instance in enumerate(instances):
        features = map(int, instance.split())
        for feature in features:
            entries.append(1)
            row.append(i)
            col.append(feature - 1)
    return sp.coo_matrix((entries, (row,col)), shape = [instances.size, 100000]).tocsr()


# In[201]:


x_sparse = transform_to_sparse(x)
test_sparse=transform_to_sparse(test)


# In[202]:


#Random Undersampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy=1)
x_resampled, y_resampled = rus.fit_resample(x_sparse, y)


# In[203]:


#Random Oversampling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(sampling_strategy=1)
x_resampled, y_resampled = ros.fit_resample(x_sparse, y)


# In[204]:


#SMOTE sampling
from imblearn.over_sampling import SMOTE
smo = SMOTE(sampling_strategy=1,k_neighbors=2)
x_resampled, y_resampled = smo.fit_resample(x_sparse, y)


# In[ ]:





# In[205]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size = 0.30, random_state = 0)


# In[206]:





# In[207]:


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=6)
x_train = svd.fit_transform(x_train)
x_test = svd.transform(x_test)

#test data
test_sparse_reduced=svd.transform(test_sparse)


# In[208]:


x_train


# In[209]:


from sklearn.linear_model import LogisticRegression
#Calculate runtime
s = time.time()
classifier = LogisticRegression(C=1,random_state = 0, class_weight={0:1, 1:2})
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
e = time.time()
print("Time (Seconds)",e - s)


# In[210]:


y_hat = classifier.predict(test_sparse_reduced)


# In[211]:


from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
ac = metrics.accuracy_score(y_test, y_pred)

print(cm)
print(f1)
print(ac)


# In[213]:


np.savetxt('./submission.txt', y_hat,fmt='%s')


# In[ ]:




