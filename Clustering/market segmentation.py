import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import statsmodels.api as sm
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing

sns.set()

data=pd.read_csv('../Data/3.12. Example.csv')

x=data.copy()

kmeans=KMeans(2)
kmeans.fit(x)

clusters=x.copy()
clusters['cluster_pred']=kmeans.fit_predict(x)

x_scaled=preprocessing.scale(x)

#the elbow method to find the best number of clusters
wcss=[]
for i in range(1,10):
    kmeans=KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)


# plot.plot(range(1,10), wcss)
# plot.xlabel('Number of clusters')
# plot.ylabel('WCSS')
# plot.show()


kmeans_new=KMeans(4)
kmeans_new.fit(x_scaled)
clusters_new=x.copy()
clusters_new['cluster_pred']=kmeans_new.fit_predict(x_scaled)
plot.scatter(clusters_new['Satisfaction'],clusters_new['Loyalty'],c=clusters_new['cluster_pred'],cmap='rainbow')
plot.xlabel('Satisfaction')
plot.ylabel('Loyalty')
plot.show()