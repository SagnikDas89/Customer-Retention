#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import os
import csv
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# In[3]:


data = pd.read_csv(r"C:\Users\SAGNIK DAS\Desktop\New folder (3)\codedsheet111.csv")
# understanding the data
data.head()


# In[4]:


data.describe()


# In[5]:


data.shape


# In[6]:


data.tail()


# In[6]:


data.info()


# In[7]:


data.columns


# In[8]:


data.nunique()


# In[9]:


data.isnull().sum()


# In[10]:


corelation = data.corr()


# In[11]:


sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns
            ,annot=True)


# In[ ]:


sns.pairplot(data)


# In[1]:


sns.relplot(x= 'Which city do you shop online from' y= 'The Convenience of patronizing the online retailer', hue='What is the Pin Code of where you shop online from', data=data)


# In[11]:


sns.boxplot


# In[12]:


x=data.iloc[:,:47].values
y=data.iloc[:,-1].values
y


# In[13]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[14]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[15]:


x_train


# In[16]:


from sklearn.decomposition import PCA
pca=PCA(n_components=None)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
pca.explained_variance_ratio_


# In[17]:


y_train


# In[18]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)


# In[19]:


x_test


# In[20]:


y_test


# In[21]:


y_pred=classifier.predict(x_test)
y_pred


# In[23]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(x_test,y_pred)
cm


# In[24]:


x_set,y_set = x_train,y_train

plt.scatter(x_set[y_set==1,0],x_set[y_set==1,1],label=1)
plt.scatter(x_set[y_set==2,0],x_set[y_set==2,1],label=2)
plt.scatter(x_set[y_set==3,0],x_set[y_set==3,1],label=3)

A1=np.arange(x_set[:,0].min()-1,x_set[:,0].max()+1,0.01)
A2=np.arange(x_set[:,1].min()-1,x_set[:,1].max()+1,0.01)

X1,X2=np.meshgrid(A1,A2)

z=classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape)
plt.contourf(X1,X2,z,alpha=0.2)


plt.legend()
plt.show()


# In[25]:


X1


# In[26]:


np.array([X1.ravel(),X2.ravel()])


# In[ ]:




