import pandas as pd
import os 
os.chdir("D:\data")
df=pd.read_csv("mobile_dataset.csv")
df.head()

#### 1.Univariate Selection
X=df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
Y=df["price_range"]
X.head()
Y.head() 

####Univariate Analysis
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#### Apply SelectKBest Algo
top_rank_feature=SelectKBest(score_func=chi2,k=20)
top_feature=top_rank_feature.fit(X,Y)
top_feature.scores_
dfScores=pd.DataFrame(top_feature.scores_,columns=["Scores"])
dfcolumns=pd.DataFrame(X.columns)
final_rank_features=pd.concat([dfcolumns,dfScores],axis=1)
final_rank_features.columns=["Features","Score"]
final_rank_features
final_rank_features.nlargest(10,"Score")

###### 2.Feature Importance 
## Gives us the score of each features of the data, the higher the score the more relivant it is
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model=ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)
top_rank=pd.Series(model.feature_importances_,index=X.columns)
top_rank.nlargest(10).plot(kind="barh")
plt.show()

####### 3. Correations
import seaborn as sns
corr=df.corr()
corr
top_features=corr.index
plt.figure(figsize=(20,20))
sns.heatmap(df[top_features].corr(),annot=True)

##removing the depending features or feature
import seaborn as sns
corr=df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]].corr()
corr
top_features=corr.index
plt.figure(figsize=(20,20))
sns.heatmap(df[top_features].corr(),annot=True)
### removing the correated features
threshold= 0.5
## funtion of removing features
def correlation(dataset, threshold): 
    col_corr = set() # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)): 
      for j in range(i): 
        if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value 
          colname = corr_matrix.columns[i]# getting the name of column
          col_corr.add(colname) 
    return col_corr

correlation(df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]],threshold)

### 4. Mutual_info_Classif ##information gain
from sklearn.feature_selection import mutual_info_classif
mutual_info=mutual_info_classif(X,Y)    
mutual_data=pd.Series(mutual_info,index=X.columns)
mutual_data.sort_values(ascending=False)
