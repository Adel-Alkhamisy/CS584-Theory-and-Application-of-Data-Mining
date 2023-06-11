#!/usr/bin/env python
# coding: utf-8

# In[753]:


import time
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import KFold
import scipy.sparse as sp


# In[754]:


train_data = pd.read_table("train.txt", header=None, skip_blank_lines=False)


# In[755]:


test_data = pd.read_table("test.txt", header=None, skip_blank_lines=False)


# In[756]:


train_data.iloc[0, :]


# In[ ]:





# In[757]:


y=np.array(train_data[0])


# In[758]:


y


# In[759]:


x=train_data[1]


# In[760]:


x


# In[761]:


test=test_data[0]
test


# In[762]:


#check for NaN values
x.isnull().values.any()
x.isnull().sum().sum()


# In[763]:


test.isnull().values.any()
test.isnull().sum().sum()


# In[764]:


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


# In[765]:


x_sparse = transform_to_sparse(x)
test_sparse=transform_to_sparse(test)


# In[766]:


test_sparse


# In[767]:


#Random Undersampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy=1)
x_resampled, y_resampled = rus.fit_resample(x_sparse, y)


# In[768]:


# #Random Oversampling
# from imblearn.over_sampling import RandomOverSampler
# ros = RandomOverSampler(sampling_strategy=1)
# x_resampled, y_resampled = ros.fit_resample(x_sparse, y)


# In[769]:


# #SMOTE sampling
# from imblearn.over_sampling import SMOTE
# smo = SMOTE(sampling_strategy=1,k_neighbors=10)
# x_resampled, y_resampled = smo.fit_resample(x_sparse, y)


# In[770]:


y_resampled


# In[771]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size = 0.30, random_state = 0)


# In[772]:


# from sklearn.decomposition import SparsePCA
# sparse_pca = SparsePCA(n_components=3)
# x_train = sparse_pca.fit_transform(x_train.toarray())
# x_test = sparse_pca.transform(x_test.toarray())

# #test data
# test_sparse_reduced=sparse_pca.transform(test_sparse.toarray())


# In[773]:


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=25)
x_train = svd.fit_transform(x_train)
x_test = svd.transform(x_test)

#test data
test_sparse_reduced=svd.transform(test_sparse)


# In[774]:


x_train


# In[775]:


from sklearn.tree import DecisionTreeClassifier
#Calculate runtime
s = time.time()
classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0,max_depth=3,class_weight={0:1,1:2})
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
e = time.time()
print("Time (Seconds)",e - s)


# In[776]:


classifier.max_depth


# In[777]:


#y_hat = classifier.predict(test_sparse)
y_hat = classifier.predict(test_sparse_reduced)


# In[778]:


from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
ac = metrics.accuracy_score(y_test, y_pred)

print(cm)
print(f1)
print(ac)


# In[779]:


np.savetxt('./submission.txt', y_hat,fmt='%s')


# In[ ]:




