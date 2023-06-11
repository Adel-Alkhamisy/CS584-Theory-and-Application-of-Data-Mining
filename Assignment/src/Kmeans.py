#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA


# In[43]:


class KMeans:
    
    def __init__(self,x,k,n_init,max_iteration):
        self.x=x
        self.k=k
        self.n_init=n_init
        self.max_iteration=max_iteration
    
    def fit_predict(self):
        x=self.x
        k=self.k
        n_init=self.n_init
        max_iteration=self.max_iteration

        rng=np.random.default_rng()
        assignment_list=[]
        variancesums=[]

        for recompute in range(n_init):
            #create random centroids
            centroids=np.zeros((k,x.shape[1]))
            for i in range(k):
                random_index=int(x.shape[0] * rng.random())
                centroids[i]=x[random_index]

            old_centroids=np.zeros((k,x.shape[1]))
            #check if the centroids changed by calculating Euclidean distance to whole old and new centriods arrays
            #if any change happend it will be greater than 0.Note: the output will be one number only
            iteration=0
            while np.linalg.norm(old_centroids-centroids)>0 and iteration<max_iteration:
                clusterPerPoint=np.zeros(x.shape[0])
                #add points to clusters
                for i in range(x.shape[0]):
                    distances = [np.linalg.norm(x[i]-centroid) for centroid in centroids]
                    nearest_centroid=np.argmin(distances)
                    clusterPerPoint[i]=nearest_centroid
                old_centroids=centroids.copy()

                centroids=np.zeros((k,x.shape[1]))
                for i in range(k):
                    centroids[i]=np.average(x[clusterPerPoint==i],axis=0)
                centroids
                iteration=iteration+1

            variancesum=0
            for centroid_index in range(k):
                variancesum +=np.mean(np.abs(x[clusterPerPoint==centroid_index]- centroids[centroid_index])**2)

            #print("variance",variancesum)
            variancesums.append(variancesum)
            assignment_list.append(clusterPerPoint.copy())

        best_clustring=assignment_list[np.argmin(variancesums)]
        worst_clustring=assignment_list[np.argmax(variancesums)]
        
        #return best_clustring,worst_clustring
        return best_clustring.astype(int)


# In[44]:


class KMeansPlusPlus:
    
    def __init__(self,x,k,n_init,max_iteration):
        self.x=x
        self.k=k
        self.n_init=n_init
        self.max_iteration=max_iteration
    
    def fit_predict(self):
        x=self.x
        k=self.k
        n_init=self.n_init
        max_iteration=self.max_iteration

        rng=np.random.default_rng()
        
        assignment_list=[]
        variancesums=[]

        for recompute in range(n_init):
            #create the first random centriod
            random_index=int(x.shape[0] * rng.random())
            centroids=np.array([x[random_index]])
            for k_ in range(1,k):
                distance=np.array([])
                for x_ in x:
                    distance=np.append(distance,np.min(np.sum((x_-centroids)**2)))
                p=distance/np.sum(distance)
                cummulative_probability=np.cumsum(p)
                r=rng.random()
                random_index=0
                for index,pr in enumerate(cummulative_probability):
                    if r<pr:
                        random_index=index
                        break
                centroids=np.append(centroids,[x[random_index]],axis=0)
            #--------------------------------------------------------------
            
            
            #--------------------------------------------------------------
            old_centroids=np.zeros((k,x.shape[1]))
            #check if the centroids changed
            iteration=0
            while np.linalg.norm(old_centroids-centroids)>0 and iteration<max_iteration:
                clusterPerPoint=np.zeros(x.shape[0])
                #add the points to clusters
                for i in range(x.shape[0]):
                    distances = [np.linalg.norm(x[i]-centroid) for centroid in centroids]
                    nearest_centroid=np.argmin(distances)
                    clusterPerPoint[i]=nearest_centroid
                old_centroids=centroids.copy()

                centroids=np.zeros((k,x.shape[1]))
                for i in range(k):
                    centroids[i]=np.average(x[clusterPerPoint==i],axis=0)
                centroids
                iteration=iteration+1

            variancesum=0
            for centroid_index in range(k):
                variancesum +=np.mean(np.abs(x[clusterPerPoint==centroid_index]- centroids[centroid_index])**2)

            #print("variance",variancesum)
            variancesums.append(variancesum)
            assignment_list.append(clusterPerPoint.copy())

        best_clustring=assignment_list[np.argmin(variancesums)]
        worst_clustring=assignment_list[np.argmax(variancesums)]
        #return best_clustring,worst_clustring
        return best_clustring.astype(int)


# In[18]:


iris = datasets.load_iris()
x = iris.data[:, :]
y = iris.target


# In[19]:


x_reduced= PCA(n_components=2).fit_transform(x)


# In[ ]:


sns.scatterplot(x=x_reduced[:,0],y=x_reduced[:,1],hue=y)


# In[21]:


kmeans=KMeans(x,3,10)
best_assignment=kmeans.fit_predict()


# In[ ]:


sns.scatterplot(x=x_reduced[:,0],y=x_reduced[:,1],hue=best_assignment)


# In[23]:


#sns.scatterplot(x=x_reduced[:,0],y=x_reduced[:,1],hue=worst_assignment)


# In[28]:


kmeansplusplus=KMeansPlusPlus(x,3,10)
best_assignment=kmeansplusplus.fit_predict()


# In[ ]:


best_assignment


# In[ ]:


sns.scatterplot(x=x_reduced[:,0],y=x_reduced[:,1],hue=best_assignment)


# In[ ]:


#sns.scatterplot(x=x_reduced[:,0],y=x_reduced[:,1],hue=worst_assignment)


# In[48]:


# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics import silhouette_samples, silhouette_score
for k in range(2,11):
    kmeans=KMeans(x,k,10)
    best_assignment=kmeans.fit_predict()
    silhouette = silhouette_score(x,best_assignment)
    v_score= v_measure_score(y,best_assignment)
    print("For n_clusters =",k,"The average silhouette_score is :", silhouette, "The V_measure_score is :", v_score,)


# In[ ]:


# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics import silhouette_samples, silhouette_score
for k in range(2,11):
    kmeans=KMeansPlusPlus(x,k,10)
    best_assignment=kmeans.fit_predict()
    silhouette = silhouette_score(x,best_assignment)
    v_score= v_measure_score(y,best_assignment)
    print("For n_clusters =",k,"The average silhouette_score is :", silhouette, "The V_measure_score is :", v_score,)


# In[36]:





# In[ ]:





# In[39]:





# In[40]:





# In[3]:


test_data = np.genfromtxt("test.txt", dtype=int, encoding=None, delimiter=",", invalid_raise=True)


# In[6]:


np.nansum(test_data)


# In[4]:


test_data.shape


# In[ ]:


test_data[0]


# In[45]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
test_data = sc.fit_transform(test_data)
#with_mean=False


# In[51]:


test_data_Reduced=PCA(n_components=15).fit_transform(test_data)


# In[37]:


test_data_Reduced.shape[1]


# In[71]:





# In[55]:


test_data_Reduced.shape


# In[46]:


from sklearn.manifold import TSNE
test_data_Reduced = TSNE(n_components=2, learning_rate='auto', init='random',perplexity=50.0).fit_transform(test_data)
#learning_rate=300.0


# In[47]:


# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters
from sklearn.metrics import silhouette_samples, silhouette_score
for k in range(2,21,2):
    kmeans2=KMeansPlusPlus(test_data_Reduced,10,5,8)
    best_assignment=kmeans2.fit_predict()
    silhouette = silhouette_score(x,best_assignment)
    print("For n_clusters =",k,"The average silhouette_score is :", silhouette, "The V_measure_score is :", v_score,)
    new_file = open('clusters'+str(k)+'.txt','w')

    for cluster in best_assignment:
        new_file.write(str(int(cluster + 1)))
        new_file.write('\n')
    new_file.close()

