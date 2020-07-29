####### Missing Value Feature Engg
## 1.MCAR (missing completly at random)
import pandas as pd
import os
os.chdir("D:\data")
df=pd.read_csv("train.csv")
df.head()
df.tail()
##finding the null values
df.isnull().sum()

## 2.Missiing data not at random (MNAR)
import numpy as np
df["cabin_null"]=np.where(df["Cabin"].isnull(),1,0)
##find the percentage of null 
df["cabin_null"].mean()
df.columns


##Analyisis
df.groupby(["Survived"])["Cabin"]
df.groupby(["Survived"])["cabin_null"].mean()

##### 3.Missing at random
## i.  Mean/Median/Mode Replacement
## ii. Random Sample IMputations
## iii.Capturing NAN values with a new feature 
## iv. End of distribution imputation
## v.  Arbitary imputation
## vi. Freq catagory imputation 

#### Mean/Median/Mode imputation
df1=pd.read_csv("train.csv",usecols=["Age","Fare","Survived"])
df1.head()
## lets see the percentage of NAN values
df1.isnull().mean()
## replaceing NAN with median+
def impute_nan(df,variable,median):
    df[variable+"_median"]=df[variable].fillna(median)
    
median=df1["Age"].median()
median
impute_nan(df1,"Age",median)
df1.head()
df1["Age"].describe()
df1["Age_median"].describe()
import seaborn as sns
sns.distplot(df1["Age_median"])
## comparing Age and Age_Median
import matplotlib.pyplot as plt
%matplotlib inline
fig = plt.figure()
ax = fig.add_subplot(111)
df['Age'].plot(kind='kde', ax=ax)
df.Age_median.plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')