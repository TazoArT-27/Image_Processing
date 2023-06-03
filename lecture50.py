
#*K Means Clustering Coding. 
#*For data based analysis Scikit Learn is good and for image based analysis OpenCV is good. 

import pandas as pd
from matplotlib import pyplot as plt

df=pd.read_excel('CSVFiles/K_Means.xlsx')
# print(df.head())
# print(df.info())

import seaborn as sns
sns.regplot(x=df['X'], y=df['Y'], fit_reg=False)
# plt.show()


from sklearn.cluster import KMeans

#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)

model = kmeans.fit(df)

predicted_values = kmeans.predict(df)


plt.scatter(df['X'], df['Y'], c=predicted_values, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', alpha=0.5)
plt.show()