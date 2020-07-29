import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
os.chdir("D:\data")
Wavelengths=pd.read_csv("TFE_0.csv")
#Distribution
sns.set_style("darkgrid")
sns.distplot(Wavelengths['Temp20'],kde = False)
sns.lineplot(x="Wavelength (nm)",y="Temp5", data=Wavelengths )
Wavelengths.corr()
correlation = Wavelengths.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')


#PCA
features = ["Temp5", 'Temp10', 'Temp15', 'Temp20','Temp25','Temp30','Temp35','Temp40','Temp45','Temp50','Temp55']
import numpy as np
X= Wavelengths.loc[:, features].values
Y= Wavelengths.loc[:,['Wavelength (nm)']].values
np.shape(X)
np.shape(Y)
#to del the NAN values of X_dataframe
X_dataframe=pd.DataFrame(X)
Y_dataframe=pd.DataFrame(Y)
X_dataframe.to_csv("X_dataframe.csv")
X_dataframe=pd.read_csv("X_dataframe.csv")

#Standardized
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X_dataframe)

#Computing Eigenvectors and Eigenvalues
#First find the covarience matrix
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
plt.figure(figsize=(8,8))
sns.heatmap(cov_mat, vmax=1, square=True,annot=True,cmap='cubehelix')

#Eigenvectors and values from covarience matrix
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

#Selecting Principal Components
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)
print('Eigenvalues in descending order:')
for i in eig_pairs:
         print(i[0])
         


#PCA
from sklearn.decomposition import PCA
pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')

from sklearn.decomposition import PCA 
sklearn_pca = PCA(n_components=2)
Y_sklearn1 = sklearn_pca.fit_transform(X_std)
print(Y_sklearn1)

Y_sklearn1=pd.DataFrame(Y_sklearn1)
Y_sklearn1.to_csv("Y_sklearn.csv")
Y_sklearn1=pd.read_csv("Y_sklearn.csv")
import seaborn as sns 
sns.set_style("darkgrid")
sns.lineplot(x="Wavelength (nm)",y="PCA2", data=Y_sklearn1 )
sns.scatterplot(x="Wavelength (nm)",y="PCA2" , data=Y_sklearn1, palette="deep" )
sns.scatterplot(x="PCA1",y="PCA2" , data=Y_sklearn1, palette="deep" )
sns.lineplot(x="PCA1",y="PCA2", data=Y_sklearn1 )


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
threedee = plt.figure().gca(projection='3d')
threedee.scatter(Y_sklearn1["PCA1"], Y_sklearn1["PCA2"], Y_sklearn1["Wavelength (nm)"])
threedee.set_xlabel('PCA1')
threedee.set_ylabel('PCA2')
threedee.set_zlabel('Wavelength (nm)')
plt.show()