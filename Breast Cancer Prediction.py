#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Prediction

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# In[202]:


import seaborn as sns


# In[2]:


data = pd.read_csv('C:/Users/Apollo/OneDrive - Apollo Hospitals Enterprise Ltd/Documents/Python Scripts/Breast Cancer data.csv')


# In[3]:


data.head()


# In[4]:


data.columns


# In[5]:


data.diagnosis.unique()


# In[8]:


data.drop(['id','Unnamed: 32'],axis=1,inplace=True)


# In[9]:


data.columns


# In[12]:


data['diagnosis']=data.diagnosis.map({'M':1,'B':0})


# In[17]:


data.diagnosis.unique()


# In[18]:


data.head()


# In[27]:


plt.hist(x='diagnosis',data=data)
plt.title("Diagnosis")


# In[28]:


data.describe()


# In[29]:


features_mean = list(data.columns[1:11])


# In[30]:


features_mean


# In[35]:



dfm=data[data.diagnosis==1]
dfb=data[data.diagnosis==0]


# In[37]:





# In[132]:


#Stacked Chart

plt.rcParams.update({'font.size':8})
fig,axes=plt.subplots(nrows=5,ncols=2,figsize=(8,10))
axes=axes.ravel()

for idx,axe in enumerate(axes):
    axe.figure          
    binwidth = (max(data[features_mean[idx]])-min(data[features_mean[idx]]))/50
    axe.hist([dfm[features_mean[idx]],dfb[features_mean[idx]]],
    bins = 30,
    alpha=0.5,stacked=True,density=True,label=["M","B"],color=['r','g'])
    axe.legend(loc='upper right')
    axe.set_title(features_mean[idx])       
             
plt.tight_layout()
plt.show()


# Observations
# mean values of cell radius, perimeter, area, compactness, concavity and concave points can be used in classification of the cancer. Larger values of these parameters tends to show a correlation with malignant tumors.
# mean values of texture, smoothness, symmetry or fractual dimension does not show a particular preference of one diagnosis over the other. In any of the histograms there are no noticeable large outliers that warrants further cleanup

# Creating Test and Train Split

# In[134]:


traindata,testdata = train_test_split(data,test_size=0.3)


# In[136]:


traindata.head()


# In[146]:


testdata.head()


# In[163]:


def classification_model(model,data,x_train,y_train):
    model.fit(data[x_train],data[y_train])
    predict=model.predict(data[x_train])
    accuracy = metrics.accuracy_score(predict,data[y_train])
    print("Accuracy is {}".format(accuracy))


# In[164]:


x_train= ['radius_mean','perimeter_mean','area_mean','compactness_mean','concavity_mean','concave points_mean']
y_train = ['diagnosis']
model=LogisticRegression()
#classification_model(model,traindata,x_train,y_train)


# In[215]:


#Decision Tree


x_train= ['radius_mean','perimeter_mean','area_mean','compactness_mean','concavity_mean','concave points_mean']
y_train = ['diagnosis']
model=DecisionTreeClassifier()
classification_model(model,traindata,x_train,y_train)


# In[216]:


x_train= ['radius_mean']
y_train = ['diagnosis']
model=DecisionTreeClassifier()
classification_model(model,traindata,x_train,y_train)


# In[217]:


#Random Forest

x_train= features_mean
y_train = ['diagnosis']
model=RandomForestClassifier(n_estimators=100,min_samples_split=25, max_depth=7, max_features=2)
classification_model(model,traindata,x_train,y_train)


# In[222]:


#Random Forest

x_train= ['radius_mean']
y_train = ['diagnosis']
model=RandomForestClassifier(n_estimators=100)
classification_model(model,traindata,x_train,y_train)


# In[218]:


corr= traindata[features_mean].corr()


# In[219]:


sns.heatmap(traindata[['radius_mean','perimeter_mean','area_mean','compactness_mean','concavity_mean','concave points_mean','diagnosis']].corr(),annot=True)


# # Using on the test data set

# In[224]:


x_test= features_mean
y_test = ['diagnosis']
model = RandomForestClassifier(n_estimators=100,min_samples_split=25, max_depth=7, max_features=2)
classification_model(model, testdata,x_test,y_test)


# # Conclusion
# The best model to be used for diagnosing breast cancer as found in this analysis is the Random Forest model with the top 6 predictors, 'concave points_mean','area_mean','radius_mean','perimeter_mean','concavity_mean'. It gives a prediction accuracy of ~97% for the test data set.
# 
# I will see if I can improve this by cross validation as well as tweaking the data.

# In[ ]:




