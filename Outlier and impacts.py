#!/usr/bin/env python
# coding: utf-8

# ### which ML models are  sensitive to outliers?
#  1. Naivye bayes classifiers----------------------not Sensitive 
#  2. SVM-------------------------------------------not Sensitive
#  
#  3.LInear Reggresion------------------------------Sensitive
#  
#  4.Logistic Reggresion----------------------------Sensitive 
#  
#  5.Decision Tree Regressor or Classifier----------not Sensitive
#  
#  6.Ensemble (Random Forest)-----------------------not Sensitive
#  
#  7.KNN--------------------------------------------not Sensitive
#  
#  8.Kmeans-----------------------------------------Sensitive
#  
# 
#  9.Hierchical Clustering -------------------------Sensitive
#  
# 10.PCA--------------------------------------------Sensitive
# 
# 11.Nerual Networks--------------------------------Sensitive

# 

# In[73]:


import pandas as pd
import os
os.chdir("D:\data")


# In[74]:


df=pd.read_csv("train.csv")


# In[75]:


df.head()


# In[76]:


df["Age"].isnull().sum()


# In[77]:


import seaborn as sns


# In[78]:


sns.distplot(df["Age"].dropna())


# In[79]:


### creating outliers
sns.distplot(df["Age"].fillna(100))


# ### if the data is Gaussian Distributed then use only upper boundary and lower boundary by 3*std 

# In[80]:


fig=df["Age"].hist(bins=50)
fig.set_title("Age")
fig.set_xlabel("Age")
fig.set_ylabel("No of passengers")


# In[81]:


sns.boxplot(df["Age"])


# In[82]:


df["Age"].describe()


# In[83]:


### Assuming Age follows Gaussian Distribution we will calculate the boundary which diffrentiate the outliers

upper_boundary=df["Age"].mean()+3*df["Age"].std()
lower_boundary=df["Age"].mean()-3*df["Age"].std()
print(upper_boundary)
print(lower_boundary)
print(df["Age"].mean())


# In[84]:


df["Age"][df["Age"]>73]=73


# In[99]:


df["Age"].hist(bins=50)


# In[30]:


### lets compute the  IOR to calcute the  boundary 
IOR = 38- 20.125
IOR


# In[31]:


max_point=(38+(1.5*(38-20.125)))
max_point


# In[34]:


min_point=(20.125-(1.5*(38-20.125)))
min_point


# In[32]:


df["Age"][df["Age"]>max_point]=max_point


# In[33]:


sns.boxplot(df["Age"])


# ### if the  data is skewed  then use IOR techique 

# In[92]:


fare_fig=df["Fare"].hist(bins=50)
fare_fig.set_title("Fare")
fare_fig.set_xlabel("Fare")
fare_fig.set_ylabel("No.of passenger")


# In[93]:


sns.boxplot(df["Fare"])


# In[94]:


df["Fare"].describe()


# ### for extreme outliers we have to take 3 or else 1.5

# In[95]:


max_point_1=(31+(3*(31-7.910)))
max_point_1


# In[96]:


min_point_1=(7.910-(3*(31-7.910)))
min_point_1


# In[97]:


df["Fare"][df['Fare']>max_point_1]=max_point_1


# In[98]:


df["Fare"].hist(bins=50)


# In[101]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df[["Age","Fare"]].fillna(0),df["Survived"],test_size=0.3)


# In[102]:


### logistic regg 
from sklearn.linear_model import LogisticRegression


# In[110]:


classifier=LogisticRegression()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
y_pred1=classifier.predict_proba(X_test)

from sklearn.metrics import accuracy_score,roc_auc_score
print("Accuracy_Score: {}".format(accuracy_score(y_test,y_pred)) )
print("roc_auc_score: {}".format(roc_auc_score(y_test,y_pred1[:,1])) )


# In[111]:


### RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
y_pred1=classifier.predict_proba(X_test)

from sklearn.metrics import accuracy_score,roc_auc_score
print("Accuracy_Score: {}".format(accuracy_score(y_test,y_pred)) )
print("roc_auc_score: {}".format(roc_auc_score(y_test,y_pred1[:,1])) )


# In[ ]:




