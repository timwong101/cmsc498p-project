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

import numpy as np
from spectral_clustering.core.util import get_scale, spectral_clustering
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

def process(x_spectralnet, y_spectralnet, data, params):
    # UNPACK DATA
    x_train, y_train, x_val, y_val, x_test, y_test = data['spectral']['train_and_test']

    # concatenate
    x = np.concatenate([x_train, x_val, x_test], axis=0)
    y = np.concatenate([y_train, y_val, y_test], axis=0)

    # PERFORM SPECTRAL CLUSTERING ON DATA

    # get eigenvalues and eigenvectors
    scale = get_scale(x, params['batch_size'], params['scale_nbr'])
    values, vectors = spectral_clustering(x, scale, params['n_nbrs'], params['affinity'])

    # sort, then store the top n_clusters=2
    values_idx = np.argsort(values)
    x_spectral_clustering = vectors[:, values_idx[:params['n_clusters']]]

    # do kmeans clustering in this subspace
    y_spectral_clustering = KMeans(n_clusters=params['n_clusters']).fit_predict(vectors[:, values_idx[:params['n_clusters']]])

    # PLOT RESULTS

    # plot spectral net clustering
    fig2 = plt.figure()
    if x.shape[1] == 2:
        ax1 = fig2.add_subplot(311)
        ax1.scatter(x[:, 0], x[:, 1],
                    alpha=0.5, s=20, cmap='rainbow', c=y_spectralnet, lw=0)
    ax1.set_title("x colored by net prediction")

    # plot spectral clustering clusters
    if x.shape[1] == 2:
        ax2 = fig2.add_subplot(313)
        ax2.scatter(x[:, 0], x[:, 1],
                    alpha=0.5, s=20, cmap='rainbow', c=y_spectral_clustering, lw=0)
    ax2.set_title("x colored by spectral clustering")

    # plot histogram of eigenvectors
    fig3 = plt.figure()
    ax1 = fig3.add_subplot(212)
    ax1.hist(x_spectral_clustering)
    ax1.set_title("histogram of true eigenvectors")
    ax2 = fig3.add_subplot(211)
    ax2.hist(x_spectralnet)
    ax2.set_title("histogram of net outputs")

    # plot eigenvectors
    y_idx = np.argsort(y)
    fig4 = plt.figure()
    ax1 = fig4.add_subplot(212)
    ax1.plot(x_spectral_clustering[y_idx])
    ax1.set_title("plot of true eigenvectors")
    ax2 = fig4.add_subplot(211)
    ax2.plot(x_spectralnet[y_idx])
    ax2.set_title("plot of net outputs")

    plt.draw()
    plt.show()


######################################################
def parallelPlot(dfp):
    parallel_coordinates(dfp, class_column='medoid', colormap=get_cmap("Set1"))
    plt.show()

def radarPlot(dfp, medoidsList, dataset):
    if dataset == 'data/mushrooms.csv':
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
    elif dataset == 'data/crime_without_states.csv' or dataset == '../data/crime_without_states.csv':
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
        plt.yticks([.1, .2, .3, .4, .5, .6, .7 , .8 , .9, 1], [".1", ".2", ".3", ".4", ".5", ".6", ".7" , ".8" , ".9", "1"], color="grey", size=7)
        plt.ylim(0, 1)
        # add plots
        for i in range(len(medoidsList)):
            medoidname = medoidsList[i].index[0]
            values = medoidsList[i].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=medoidname)
            ax.fill(angles, values, 'b', alpha=0.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.show()

def facetedRadarPlot(dfp, medoidsList, dataset):
    if dataset == 'data/mushrooms.csv':
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
    elif dataset == 'data/crime_without_states.csv' or dataset == '../data/crime_without_states.csv':
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
            plt.yticks([.1, .2, .3, .4, .5, .6, .7 , .8 , .9, 1], [".1", ".2", ".3", ".4", ".5", ".6", ".7" , ".8" , ".9", "1"], color="grey", size=7)
            plt.ylim(0, 1)
            medoidname = medoidsList[row].index[0]
            values = medoidsList[row].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=medoidname)
            ax.fill(angles, values, 'b', alpha=0.1)
            plt.title(title, size=11, color=color, y=1.1)

        my_dpi = 96
        plt.figure(figsize=(1000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
        my_palette = get_cmap("Set2", len(medoidsList))

        # for each medoid call make_spider to create a radar plot
        for row in range(len(medoidsList)):
            make_spider(row=row, title=medoidsList[row].index[0], color=my_palette(row))
        plt.show()

def silhouetteAnalysis(dfp, k, centers):
    labels = []
    for i in range(len(dfp)):
        labels.append(int(dfp.iloc[i]['medoid']))
    npLabels = np.asarray(labels, dtype = np.int32)
    dfpNoMedoids = dfp.drop('medoid', 1).values
    avgSilhouetteScore = silhouette_score(dfpNoMedoids, npLabels)
    print("Average Silhouette Score: ", avgSilhouetteScore)
    sample_silhouette_values = silhouette_samples(dfpNoMedoids, npLabels)

    fig, ax = plt.subplots()
    fig.set_size_inches(18, 7)
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(dfpNoMedoids) + (k + 1) * 10])
    y_lower = 10
    color_num = 0
    for i in centers:
        color_num = color_num + 1
        ith_cluster_silhouette_values = sample_silhouette_values[npLabels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(color_num) / k)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=avgSilhouetteScore, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()

    return avgSilhouetteScore
