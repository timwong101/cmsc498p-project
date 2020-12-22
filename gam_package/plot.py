from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from math import pi
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def parallelPlot(dfp):
    parallel_coordinates(dfp, class_column='medoid', colormap=get_cmap("Set1"))
    plt.show()

def radarPlot(dfp, medoidsList):
    # create background
    col = dfp.pop('medoid')
    dfp.insert(0, 'medoid', col)
    categories = list(dfp)[1:]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([.1, .2, .3], [".1", ".2", ".3"], color="grey", size=7)
    plt.ylim(0, .3)
    # add plots
    for i in range(len(medoidsList)):
        medoidname = medoidsList[i].index[0]
        values = medoidsList[i].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=medoidname)
        ax.fill(angles, values, 'b', alpha=0.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.show()

def facetedRadarPlot(dfp, medoidsList):
    # creates a radar plot for the given medoid
    def make_spider(row, title, color):
        col = dfp.pop('medoid')
        dfp.insert(0, 'medoid', col)
        categories = list(dfp)[1:]
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        ax = plt.subplot(2, 2, row + 1, polar=True, )
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories, color='grey', size=8)
        ax.set_rlabel_position(0)
        plt.yticks([.1, .2, .3], [".1", ".2", ".3"], color="grey", size=7)
        plt.ylim(0, .3)
        medoidname = medoidsList[row].index[0]
        values = medoidsList[row].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=medoidname)
        ax.fill(angles, values, 'b', alpha=0.1)
        plt.title(title, size=11, color=color, y=1.1)

    my_dpi = 96
    plt.figure(figsize=(1000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
    my_palette = get_cmap("Set2", len(medoidsList))

    #for each medoid call make_spider to create a radar plot
    for row in range(len(medoidsList)):
        make_spider(row=row, title=medoidsList[row].index[0], color=my_palette(row))
    plt.show()

def silhouetteAnalysis(dfp, mlist, k):
    X, y = make_blobs(n_samples=500,
                      n_features=2,
                      centers=4,
                      cluster_std=1,
                      center_box=(-10.0, 10.0),
                      shuffle=True,
                      random_state=1)
    clusterer = KMeans(n_clusters=3, random_state=10)
    cluster_labels = clusterer.fit_predict(X)


    labels = []
    for i in range(len(dfp)):
        labels.append(int(dfp.iloc[i]['medoid']))
    npLabels = np.asarray(labels, dtype = np.int32)
    dfpNoMedoids = dfp.drop('medoid', 1).values
    medoidslist = []
    for i in range(len(mlist)):
        medoidslist.append(mlist[i].index[0])
    avgSilhouetteScore = silhouette_score(dfpNoMedoids, npLabels)
    print("Average Silhouette Score: ", avgSilhouetteScore)
    sample_silhouette_values = silhouette_samples(dfpNoMedoids, npLabels)

    fig, ax = plt.subplots()
    fig.set_size_inches(18, 7)
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(dfpNoMedoids) + (k+1)*10])
    y_lower = 10
    color_num = 0
    for i in medoidslist:
        color_num = color_num + 1
        ith_cluster_silhouette_values = sample_silhouette_values[npLabels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(color_num) / k)
        ax.fill_betweenx(np.arrange(y_lower,y_upper), 0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha = 0.7)



    return avgSilhouetteScore

if __name__ == '__main__':


    print(__doc__)

    # Generating the sample data from make_blobs
    # This particular setting has one distinct cluster and 3 clusters placed close
    # together.
    X, y = make_blobs(n_samples=500,
                      n_features=2,
                      centers=4,
                      cluster_std=1,
                      center_box=(-10.0, 10.0),
                      shuffle=True,
                      random_state=1)  # For reproducibility

    range_n_clusters = [2, 3, 4, 5, 6]

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

    plt.show()