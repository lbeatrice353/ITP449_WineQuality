""" Bebe Lin
    ITP-449
    HW12
    This code analyzes winequalityreds.csv and groups them into several clusters
"""
# importing packages needed for this assignment
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

def main():
    # loading in the data
    file_path = 'wineQualityReds.csv'
    df_wines_red = pd.read_csv(file_path)

    # selecting attributes and records
    quality_series = df_wines_red['quality']
    df_wines_red = df_wines_red.drop(columns='quality')

    # transforming the data
    X = pd.DataFrame(Normalizer().fit_transform(df_wines_red), columns=df_wines_red.columns)

    # train the model:
    # pick a range of number of clusters
    # for each cluster number:
        # train a model
        # calculate silhouette score and save it somewhere
    clusters_range = range(1, 11)
    inertia_values = []
    for cluster_number in clusters_range:
        kmeans_model = KMeans(n_clusters=cluster_number, random_state=42)
        kmeans_model.fit(X)
        inertia_values.append(kmeans_model.inertia_)

    # creating a plot and labeling the axes
    plt.plot(clusters_range, inertia_values, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Wine Quality: Inertia vs Number of Clusters')
    plt.savefig('inertia_plot.png')
    plt.show()

    # finding the optimal k value(number of clusters)
    optimal_k = 2

    # clustering the wines a again into the optimal number of k clusters
    kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans_optimal.fit_predict(X)

    # extracting the cluster numbers from the model and combining with the quality
    results = pd.DataFrame({'cluster': clusters, 'quality': quality_series})

    # showing the crosstab of cluster number vs quality
    crosstab_result = pd.crosstab(results['cluster'], results['quality'])
    print(crosstab_result)

if __name__ == '__main__':
    main()



