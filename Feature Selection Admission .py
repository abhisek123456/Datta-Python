#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os 
import pandas as pd


# In[3]:


os.chdir("D:\data\Admission")
df=pd.read_csv("Admission_Predict_Ver1.1.csv")


# In[4]:


df.head()


# In[5]:


df.isnull().sum()


# In[6]:


X=df.iloc[:,[1,2,4,5,6]]
X.head()


# In[12]:


y=df.iloc[:,[3]]
y=pd.DataFrame(y)
y.tail()


# In[8]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[14]:


top_ranked_feature=SelectKBest(score_func=chi2,k=5)
results=top_ranked_feature.fit(X,y)


# In[16]:


results.scores_


# In[19]:


dfscores=pd.DataFrame(results.scores_,columns=["Scores"])
dfscores


# In[21]:


dfcolumns=pd.DataFrame(X.columns,columns=["Features"])
dfcolumns


# In[28]:


dftop_features=pd.concat([dfcolumns,dfscores],axis=1)
dftop_features.columns=["Features","Scores"]
dftop_features.nlargest(5,"Scores")


# In[32]:


import seaborn as sns 
import matplotlib.pyplot as plt
corr=df.corr()
corr


# In[37]:


top_features=corr.index
plt.figure(figsize=(20,20))
sns.heatmap(df[top_features].corr(),annot=True)


# In[44]:


threshold= 0.4


# In[42]:


def correlation(dataset, threshold): 
    col_corr = set() # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)): 
      for j in range(i): 
        if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value 
          colname = corr_matrix.columns[i]# getting the name of column
          col_corr.add(colname) 
    return col_corr


# In[46]:


correlation(df.iloc[:,[1,2,4,5,6]],threshold)


# In[47]:


from sklearn.feature_selection import mutual_info_classif
mutual_info=mutual_info_classif(X,y)    
mutual_data=pd.Series(mutual_info,index=X.columns)
mutual_data.sort_values(ascending=False)

