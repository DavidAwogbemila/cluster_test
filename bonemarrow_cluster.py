#ya29.GlvwBG8tttzkPCY7CneeT_freiA8P8nYshrBEqJDa-3gC7JiwZq7QAl7Yp4qZOo0F9H3rY0Dvr9YxgyWxoL01twka2uw3_OZ97IddHbjJ5TiPoR_ljv_du6Dd0EP
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics

bonemarrow_df_full = pd.read_csv("bonemarrow_training.csv", sep=",", header=0, nrows=1000)
brain_df_full = pd.read_csv("brain_training.csv", sep=",", header=0, nrows=1000)

df_full = pd.concat([bonemarrow_df_full, brain_df_full])
df_full.fillna(0, inplace=True)
#df_full.dropna(inplace=True)

df_no_genomic_identifier = df_full.drop("GenomicIdentifier", axis=1)

n_clust = [i for i in range(2, 11)]
sil_scores = []
calinski_scores = []

X = df_no_genomic_identifier.as_matrix()

#silhouette accuracy testing
#use 2 to 10 clusters
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
    sil_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')
    sil_scores.append(sil_score)
#results suggest use of 2 to 3 clusters

plt.figure(1)
plt.plot(n_clust, sil_scores,'k.', markersize=2)
plt.title('K-means clustering on bonemarrow and brain cancer samples.\n Measuring accuracy with silhouette score.')
plt.xlabel('number of clusters used')
plt.ylabel('silhouette score')
plt.show()

#calinski accuracy testing
#use 2 to 10 clusters
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
    calinski_score = metrics.calinski_harabaz_score(X, kmeans.labels_)
    calinski_scores.append(calinski_score)
#results suggest use of 2 to 3 clusters


plt.figure()
plt.plot(n_clust, calinski_scores,'k.', markersize=2)
plt.title('K-means clustering on bonemarrow and brain cancer samples.\n Measuring accuracy with calinski score.')
plt.xlabel('number of clusters used')
plt.ylabel('calinski score')
plt.show()

print(sil_score)
