#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import io
import requests
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
from pandas.api.types import CategoricalDtype

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


train_data = pd.read_csv('train.csv', header=0,skip_blank_lines=True)
train_data = train_data.dropna()
x=train_data.iloc[:,0:13]
y=train_data.iloc[:,-1]
test_data = pd.read_csv('test.csv', header=0,skip_blank_lines=True)
test_data = test_data.dropna()


# In[38]:


x.info()


# In[5]:


def preprocess(data,target):
    
    #education is the same as education_num, so I drop it
    data=data.drop(columns=['education'],axis=1)
    data=data.drop(columns=['native-country'],axis=1)
    data=data.drop(columns=['relationship'],axis=1)
    data=data.drop(columns=['marital-status'],axis=1)
    
    #check features list
    # data.info()
    
    #check for null values in the entire dataframe
    data.isnull().values.any()
    
    #show correlation between featues
    #create an array like data dataframe
    # mask=np.zeros_like(data.corr())
    
    #Take the top triangle indices from the mask array
#     triangle_indices=np.triu_indices_from(mask)
#     mask[triangle_indices]=True
    
#     plt.figure(figsize=(20,10))
#     sns.heatmap(data.corr(), mask=mask, annot=True,cmap="YlGnBu")
#     plt.show()
    
    #select all the numerical attributes using selec_dtypes()
    numerical_attributes = data.select_dtypes(include=['int'])
    # print(numerical_attributes.columns)
    
    # plots features histograms to see the distribution of each feature
    numerical_attributes.hist(figsize=(10,10))
    
    #select object type only which represent text data
    categorical_attributes = data.select_dtypes(include=['object'])
    print(categorical_attributes.columns)
    
    
    #replace categorical_attribute value "?" with mode (most frequent item in the column)
    for column in categorical_attributes.columns:
        categorical_attributes.loc[:,column]=categorical_attributes.loc[:,(column)].str.replace('?', categorical_attributes.loc[:,column].mode()[0])
    #------------------------
    # categorical_attributes['occupation'].replace(' ?', np.NaN, inplace=True)
    # categorical_attributes.workclass.value_counts()
    #------------------------
    
    #replace missing categorical_attributes values with mode(most frequent item in the column) in each column
    for column in categorical_attributes.columns:
        categorical_attributes.loc[:,column].fillna(categorical_attributes.loc[:,column].mode()[0],inplace=True)
    
    #replace missing numerical_attributes values with mode(most frequent item in the column) in each column
    for column in numerical_attributes.columns:
        numerical_attributes.loc[:,column].fillna(numerical_attributes.loc[:,column].mode()[0],inplace=True)
        
    #Show the counts of observations in each categorical bin using bars
#     sns.set(rc={'figure.figsize':(10,10)})
#     sns.countplot(y='workclass', hue=target, data = categorical_attributes)
    
#     sns.set(rc={'figure.figsize':(10,10)})
#     sns.countplot(y='marital-status', hue=y, data = categorical_attributes)
#     plt.show()
#     sns.set(rc={'figure.figsize':(10,10)})
#     sns.countplot(y='occupation', hue=y, data = categorical_attributes)
#     plt.show()
#     sns.set(rc={'figure.figsize':(10,10)})
#     sns.countplot(y='relationship', hue=y, data = categorical_attributes)
#     plt.show()
#     sns.set(rc={'figure.figsize':(10,10)})
#     sns.countplot(y='race', hue=y, data = categorical_attributes)
#     plt.show()
#     sns.set(rc={'figure.figsize':(10,10)})
#     sns.countplot(y='sex', hue=y, data = categorical_attributes)
#     plt.show()
#     sns.set(rc={'figure.figsize':(10,10)})
    # sns.countplot(y='native-country', hue=y, data = categorical_attributes)
    # plt.show()
    
    #Scale numerical data
    sc = StandardScaler()
    numerical_attributes = pd.DataFrame(sc.fit_transform(numerical_attributes),columns = numerical_attributes.columns)
    
    #encode sex Male=0, Female=1
    categorical_attributes=categorical_attributes.replace(to_replace=['Male','Female'],value=[0,1],regex=True)
    
    #encode categorical data
    categorical_attributes=pd.get_dummies(categorical_attributes, columns=["workclass","occupation","race"])
    
    #encode categorical data 
    # categorical_attributes=pd.get_dummies(categorical_attributes, columns=["workclass","marital-status","occupation","relationship","race"])
    # print(categorical_attributes)
    
    #concatenate numerical_attributes dataframe with categorical_attributes dataframe
    data_=pd.concat((numerical_attributes,categorical_attributes),axis=1)
    # print(data_)
    #,numerical_attributes,categorical_attributes
     
    return data_


# In[6]:


#encode '<=50K'=0 and '>50K'=1
y=y.replace(to_replace=['<=50K','>50K'],value=[0,1],regex=True)


# In[7]:


x_preprocced=preprocess(x,y)
test_preprocced=preprocess(test_data,y)


# In[13]:


x_preprocced.info()


# In[45]:


# x_preprocced=x_preprocced.drop(columns=['native-country_ Holand-Netherlands'],axis=1)
#x_preprocced=x_preprocced.drop(columns=['native-country'],axis=1)


# In[46]:


# #Random Oversampling
# from imblearn.over_sampling import RandomOverSampler
# ros = RandomOverSampler(sampling_strategy=1)
# x_preprocced, y = ros.fit_resample(x_preprocced, y)


# In[47]:


# #Random Undersampling
# from imblearn.under_sampling import RandomUnderSampler
# rus = RandomUnderSampler(sampling_strategy=1)
# x_preprocced, y = rus.fit_resample(x_preprocced, y)


# In[48]:


# x_preprocced= PCA(n_components=30).fit_transform(x_preprocced)
# test_preprocced=PCA(n_components=30).fit_transform(test_preprocced)


# In[ ]:





# In[14]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_preprocced, y, test_size = 0.35, random_state = 0)


# In[10]:


# sex_attr=x_test['sex']
# race_attr=x_test.iloc[:, 43:48]


# In[15]:


sex_attr=x_test['sex']
race_attr=x_test.iloc[:, 30:35]


# In[17]:


x_train=x_train.drop(columns=['sex'],axis=1)
x_test=x_test.drop(columns=['sex'],axis=1)
test_preprocced=test_preprocced.drop(columns=['sex'],axis=1)

x_train=x_train.drop(columns=['race_ White'],axis=1)
x_test=x_test.drop(columns=['race_ White'],axis=1)
test_preprocced=test_preprocced.drop(columns=['race_ White'],axis=1)


x_train=x_train.drop(columns=['race_ Black'],axis=1)
x_test=x_test.drop(columns=['race_ Black'],axis=1)
test_preprocced=test_preprocced.drop(columns=['race_ Black'],axis=1)

x_train=x_train.drop(columns=['race_ Amer-Indian-Eskimo'],axis=1)
x_test=x_test.drop(columns=['race_ Amer-Indian-Eskimo'],axis=1)
test_preprocced=test_preprocced.drop(columns=['race_ Amer-Indian-Eskimo'],axis=1)

x_train=x_train.drop(columns=['race_ Asian-Pac-Islander'],axis=1)
x_test=x_test.drop(columns=['race_ Asian-Pac-Islander'],axis=1)
test_preprocced=test_preprocced.drop(columns=['race_ Asian-Pac-Islander'],axis=1)

x_train=x_train.drop(columns=['race_ Other'],axis=1)
x_test=x_test.drop(columns=['race_ Other'],axis=1)
test_preprocced=test_preprocced.drop(columns=['race_ Other'],axis=1)



# In[18]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=0.5, random_state = 0,max_iter=300)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)


from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
ac = metrics.accuracy_score(y_test, y_pred)

print(cm)
print('F1_score: ',f1)
print('Accuracy: ',ac)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#test data prediction
y_hat_1=classifier.predict(test_preprocced)

#save result in text file
np.savetxt('./LR.txt', y_hat_1,fmt='%s')


# In[ ]:





# In[54]:


#decision tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0,max_depth=10)
classifier.fit(x_train, y_train)

#prediction
y_pred = classifier.predict(x_test)

from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
ac = metrics.accuracy_score(y_test, y_pred)

print(cm)
print('F1_score: ',f1)
print('Accuracy: ',ac)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#test data prediction
y_hat_2=classifier.predict(test_preprocced)

#save result in text file
np.savetxt('./DT.txt', y_hat_2,fmt='%s')


# In[55]:


#random forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000,max_depth=20)
classifier.fit(x_train, y_train)

#prediction
y_pred = classifier.predict(x_test)

from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
ac = metrics.accuracy_score(y_test, y_pred)

print(cm)
print('F1_score: ',f1)
print('Accuracy: ',ac)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#test data prediction
y_hat_3=classifier.predict(test_preprocced)

#save result in text file
np.savetxt('./RF.txt', y_hat_3,fmt='%s')


# In[28]:


from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier(n_estimators=300)
classifier.fit(x_train, y_train)

#prediction
y_pred = classifier.predict(x_test)

from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
ac = metrics.accuracy_score(y_test, y_pred)

print(cm)
print('F1_score: ',f1)
print('Accuracy: ',ac)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#test data prediction
y_hat_3=classifier.predict(test_preprocced)

#save result in text file
np.savetxt('./AdaBoost.txt', y_hat_3,fmt='%s')


# In[57]:


from sklearn.neighbors import KNeighborsClassifier

#for k in range(1,300,5):
classifier = KNeighborsClassifier(n_neighbors=55)
classifier.fit(x_train, y_train)

#prediction
y_pred = classifier.predict(x_test)
    
#print('n_neighbors: ',k)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#test data prediction
y_hat_5=classifier.predict(test_preprocced)

#save result in text file
np.savetxt('./KNN.txt', y_hat_5,fmt='%s')


# In[58]:


#SVM
from sklearn.svm import SVC
classifier = SVC(C=0.5, kernel = 'rbf', random_state = 0)
unfiar_model=classifier.fit(x_train, y_train)

#prediction
y_pred = classifier.predict(x_test)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#test data prediction
y_hat_4=classifier.predict(test_preprocced)

#save result in text file
np.savetxt('./SVM.txt', y_hat_4,fmt='%s')


# In[20]:


x_test.info()


# In[ ]:





# In[29]:


#Compute Independence/ Demographic parity
s=sex_attr.values
# s=x_test['sex'].values
male_count=0
female_count=0

for i in range(len(s)):
    if (y_pred[i]==1 and s[i]==0):
        male_count+=1
    if (y_pred[i]==1 and s[i]==1):
        female_count+=1
#comupte P(C=1|S=0)
Pr_C1_S0=(male_count/sex_attr.value_counts()[0])
# Pr_C1_S0=(male_count/x_test['sex'].value_counts()[0])
print('P(C=1|S=0)=',Pr_C1_S0)

#comupte P(C=1|S=1)
Pr_C1_S1=(female_count/sex_attr.value_counts()[1])
# Pr_C1_S1=(female_count/x_test['sex'].value_counts()[1])
print('P(C=1|S=1)=',Pr_C1_S1)

# #Average
# print('Average = ',(Pr_C1_S0+Pr_C1_S1)/2)
#Demographic Disparity Disparity DP = |P(C=1|S=0) - P(C=1|S=1)|
DP=abs(Pr_C1_S0-Pr_C1_S1)
print('Demographic Disparity = ',DP)


# In[30]:


#Compute Equality of opportunity
s=sex_attr.values
# s=x_test['sex'].values

#convert series to numpy array for easy processing
y_test_true=y_test.to_numpy()

male_count=0
female_count=0
for i in range(len(s)):
    if (y_pred[i]==1 and s[i]==0):
        if y_test_true[i]==1:
            male_count+=1
    if (y_pred[i]==1 and s[i]==1):
        if y_test_true[i]==1:
            female_count+=1
        
#comupte P(C=1| Y=1, S=0)
Pr_C1_Y1_S0=(male_count/sex_attr.value_counts()[0])
# Pr_C1_Y1_S0=(male_count/x_test['sex'].value_counts()[0])
print('P(C=1| Y=1, S=0)=',Pr_C1_Y1_S0)

#comupte P(C=1| Y=1, S=1)
Pr_C1_Y1_S1=(female_count/sex_attr.value_counts()[1])
# Pr_C1_Y1_S1=(female_count/x_test['sex'].value_counts()[1])
print('P(C=1|Y=1, S=1)=',Pr_C1_Y1_S1)
#Average
print('Average Equality of Opportunity = ', (Pr_C1_Y1_S0+Pr_C1_Y1_S1)/2)
#compute Equality of Opportunity difference as eod = |P(C=1| Y=1, S=0)= - P(C=1|Y=1, S=1)|
eod=abs(Pr_C1_Y1_S0-Pr_C1_Y1_S1)
print('Equality of Opportunity difference',eod)


# In[31]:


#Compute Equality of odds
s=sex_attr.values
# s=x_test['sex'].values

#convert series to numpy array for easy processing
y_test_true=y_test.to_numpy()

male_count=0
female_count=0
for i in range(len(s)):
    #false positive rate
    if (y_pred[i]==1 and s[i]==0):
        if y_test_true[i]==0:
            male_count+=1
    if (y_pred[i]==1 and s[i]==1):
        if y_test_true[i]==0:
            female_count+=1
        
#comupte P(C=1| Y=0, S=0)
Pr_C1_Y0_S0=(male_count/sex_attr.value_counts()[0])
# Pr_C1_Y0_S0=(male_count/x_test['sex'].value_counts()[0])
print('P(C=1| Y=0, S=0)=',Pr_C1_Y1_S0)

#comupte P(C=1| Y=1, S=1)
Pr_C1_Y0_S1=(female_count/sex_attr.value_counts()[1])
# Pr_C1_Y0_S1=(female_count/x_test['sex'].value_counts()[1])
print('P(C=1|Y=0, S=1)=',Pr_C1_Y0_S1)
#Average
print('Average Equality of Odds = ', (Pr_C1_Y0_S0+Pr_C1_Y0_S1)/2)

#compute Equality of odds difference as EoOD = |P(C=1| Y=0, S=0)= - P(C=1|Y=0, S=1)|
EoOD=abs(Pr_C1_Y0_S0-Pr_C1_Y0_S1)
print('Equality of odds difference',EoOD)


# In[32]:


race_attr.value_counts()


# In[33]:


#Compute Independence/ Demographic parity
#Reconstruct categorical variable race from dummies
x_test_race=race_attr.idxmax(axis=1)
x_test_race=x_test_race.replace(to_replace=['race_ White','race_ Black','race_ Amer-Indian-Eskimo','race_ Asian-Pac-Islander','race_ Other'],value=[0,1,1,1,1],regex=True)
race=x_test_race.to_numpy()

#Reconstruct categorical variable race from dummies
# x_test_race=x_test.iloc[:, 43:48]
# x_test_race=x_test_race.idxmax(axis=1)
# x_test_race=x_test_race.replace(to_replace=['race_ White','race_ Black','race_ Amer-Indian-Eskimo','race_ Asian-Pac-Islander','race_ Other'],value=[0,1,1,1,1],regex=True)
# race=x_test_race.to_numpy()

White_count=0
Other_count=0

for i in range(len(race)):
    if (y_pred[i]==1 and race[i]==0):
        White_count+=1
    if (y_pred[i]==1 and race[i]==1):
        Other_count+=1


#comupte P(C=1|race=0)
Pr_C1_race0 =((White_count)/(x_test_race.value_counts()[0]))
print('P(C=1|race=0)=',Pr_C1_race0)

#comupte P(C=1|race=1)
Pr_C1_race1 =((Other_count)/(x_test_race.value_counts()[1]))
print('P(C=1|race=1)=',Pr_C1_race1)


#Demographic Disparity DP = |P(C=1|race=0) - P(C=1|race=1)|
DP=abs(Pr_C1_race0-Pr_C1_race1)
print('Demographic Disparity = ',DP)


# In[ ]:





# In[67]:


len(x_test_race)


# In[34]:


#Compute equality of opportunity

#convert series to numpy array for easy processing
y_test_true=y_test.to_numpy()

x_test_race=race_attr.idxmax(axis=1)
x_test_race=x_test_race.replace(to_replace=['race_ White','race_ Black','race_ Amer-Indian-Eskimo','race_ Asian-Pac-Islander','race_ Other'],value=[0,1,1,1,1],regex=True)
race=x_test_race.to_numpy()

# #Reconstruct categorical variable race from dummies
# x_test_race=x_test.iloc[:, 43:48]
# x_test_race=x_test_race.idxmax(axis=1)
# x_test_race=x_test_race.replace(to_replace=['race_ White','race_ Black','race_ Amer-Indian-Eskimo','race_ Asian-Pac-Islander','race_ Other'],value=[0,1,1,1,1],regex=True)
# race=x_test_race.to_numpy()

White_count=0
Other_count=0

for i in range(len(race)):
    if (y_pred[i]==1 and race[i]==0):
        if y_test_true[i]==1:
            White_count+=1
    if (y_pred[i]==1 and race[i]==1):
        if y_test_true[i]==1:
            Other_count+=1


#comupte P(C=1|race=0)
Pr_C1_Y1_race0 =((White_count)/(x_test_race.value_counts()[0]))
print('P(C=1|Y=1,race=0)=',Pr_C1_Y1_race0)

#comupte P(C=1|race=1)
Pr_C1_Y1_race1 =((Other_count)/(x_test_race.value_counts()[1]))
print('P(C=1|Y=1,race=1)=',Pr_C1_Y1_race1)

#Average
print('Average Equality Of Opportunity = ',(Pr_C1_Y1_race0+Pr_C1_Y1_race1)/2)
#compute Equality of Opportunity difference as e =|P(C=1|y=1,race=0)-P(C=1|y=1,race=1)
e=abs(Pr_C1_Y1_race0-Pr_C1_Y1_race1)
print('Equality of Opportunity difference',e)


# In[35]:


#Compute equality of odds

#convert series to numpy array for easy processing
y_test_true=y_test.to_numpy()

x_test_race=race_attr.idxmax(axis=1)
x_test_race=x_test_race.replace(to_replace=['race_ White','race_ Black','race_ Amer-Indian-Eskimo','race_ Asian-Pac-Islander','race_ Other'],value=[0,1,1,1,1],regex=True)
race=x_test_race.to_numpy()

# #Reconstruct categorical variable race from dummies
# x_test_race=x_test.iloc[:, 43:48]
# x_test_race=x_test_race.idxmax(axis=1)
# x_test_race=x_test_race.replace(to_replace=['race_ White','race_ Black','race_ Amer-Indian-Eskimo','race_ Asian-Pac-Islander','race_ Other'],value=[0,1,1,1,1],regex=True)
# race=x_test_race.to_numpy()

White_count=0
Other_count=0


for i in range(len(race)):
    if (y_pred[i]==1 and race[i]==0):
        if y_test_true[i]==0:
            White_count+=1
    if (y_pred[i]==1 and race[i]==1):
        if y_test_true[i]==0:
            Other_count+=1


#comupte P(C=1|race=0)
Pr_C1_Y0_race0 =((White_count)/(x_test_race.value_counts()[0]))
print('P(C=1|Y=0,race=0)=',Pr_C1_Y0_race0)

#comupte P(C=1|race=1)
Pr_C1_Y0_race1 =((Other_count)/(x_test_race.value_counts()[1]))
print('P(C=1|Y=0,race=1)=',Pr_C1_Y0_race1)


#Average
print('Average Equality Of odds = ',(Pr_C1_Y1_race0+Pr_C1_Y1_race1)/2)

#compute Equality of odds difference as EoO =|P(C=1|y=1,race=0)-P(C=1|y=1,race=1)|
EoO=abs(Pr_C1_Y0_race0-Pr_C1_Y0_race1)
print('Equality of odds difference',EoO)


# In[ ]:




